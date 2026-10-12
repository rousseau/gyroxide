//
// tallinen_ref: instrumented copy of Tuomas Tallinen's Brains.cpp
// (Nature Physics 2016, Brain_code_and_meshes/Brains.cpp), used to produce
// reference traces for the gyroxide Rust+Burn simulator.
//
// The mechanics (growth, elasticity, contact, midplane, integration) are copied
// expression by expression from the original so results match it exactly when
// both run single-threaded. Differences with the original are limited to:
//   - command-line parameters instead of hard-coded file name and scenario,
//   - optional coordinate normalisation (the transform of the commented
//     "geom_norm.mesh" block of the original, also used by BrainGrowth),
//   - large per-node arrays on the heap instead of the stack,
//   - no OpenMP: the original accumulates forces from parallel loops without
//     atomics (data races), so it is compiled single-threaded here,
//   - NPY / CSV / JSON outputs instead of POV-Ray and Amira files.
//
// Build: g++ -O2 -std=c++17 eig3.cpp tallinen_ref.cpp -o tallinen_ref
// (no -ffast-math / -Ofast: keep IEEE semantics so traces are reproducible)
//
#include "vema.h"
#include "eig3.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <sys/stat.h>
using namespace std;

class Tetra{
public:
	int n1, n2, n3, n4;
	Matrix G; // Growth tensor
	Tetra(): n1(0), n2(0), n3(0), n4(0) {}
	Tetra(int n1_, int n2_, int n3_, int n4_): n1(n1_), n2(n2_), n3(n3_), n4(n4_) {}
};
class Face{
public:
	int n1, n2, n3;
	Face(): n1(0), n2(0), n3(0) {}
	Face(int n1_, int n2_, int n3_): n1(n1_), n2(n2_), n3(n3_) {}
};

void Eigensystem(Matrix A, Matrix& V, double d[3]);
void createNNLtriangle(vector<int>*, Vector*, vector<Face>&, int*, int, int, double, double, double);
Vector closestPointTriangle(Vector&, Vector&, Vector&, Vector&, double&, double&, double&);
void dist2surf(Vector*, int*, int, int, int*, double*);

// ---------------------------------------------------------------- outputs

static void mkdirs(const string& path) {
	string cur;
	stringstream ss(path);
	string part;
	if (!path.empty() && path[0] == '/') cur = "/";
	while (getline(ss, part, '/')) {
		if (part.empty()) continue;
		cur += part + "/";
		mkdir(cur.c_str(), 0755);
	}
}

// Minimal NPY v1.0 writer (C order, little endian).
static void write_npy(const string& path, const void* data, const char* descr, size_t itemsize, const vector<size_t>& shape) {
	string shp = "(";
	size_t count = 1;
	for (size_t k = 0; k < shape.size(); k++) {
		shp += to_string(shape[k]);
		shp += (shape.size() == 1 || k + 1 < shape.size()) ? "," : "";
		if (k + 1 < shape.size()) shp += " ";
		count *= shape[k];
	}
	shp += ")";
	string header = string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
	size_t total = 10 + header.size() + 1;
	size_t pad = (64 - total % 64) % 64;
	header += string(pad, ' ');
	header += '\n';
	ofstream out(path, ios::binary);
	const char magic[] = "\x93NUMPY";
	out.write(magic, 6);
	char ver[2] = {1, 0};
	out.write(ver, 2);
	uint16_t hlen = (uint16_t)header.size();
	out.write((const char*)&hlen, 2);
	out.write(header.data(), header.size());
	out.write((const char*)data, count * itemsize);
}
static void write_vec3(const string& path, const Vector* v, size_t n) {
	vector<double> buf(3 * n);
	for (size_t i = 0; i < n; i++) { buf[3*i] = v[i].x; buf[3*i+1] = v[i].y; buf[3*i+2] = v[i].z; }
	write_npy(path, buf.data(), "<f8", 8, {n, 3});
}
static void write_f64(const string& path, const double* v, size_t n) { write_npy(path, v, "<f8", 8, {n}); }
static void write_i32(const string& path, const int* v, vector<size_t> shape) { write_npy(path, v, "<i4", 4, shape); }

static string fmt_step(int step) { char b[32]; snprintf(b, sizeof b, "step_%07d", step); return b; }

// ---------------------------------------------------------------- scenarios

struct Preset {
	string name;
	double rho, dt_factor, t0;
	int kind; // 0: Tallinen folding (commented in original), 1: Tallinen fold+unfold (active in original), 2: BrainGrowth
};

static bool get_preset(const string& n, Preset& p) {
	if (n == "tallinen2016")        { p = {n, 0.0025, 0.025, 0.0, 0}; return true; }
	if (n == "tallinen2016-unfold") { p = {n, 0.0025, 0.025, -0.25, 1}; return true; }
	if (n == "braingrowth")         { p = {n, 0.01, 0.05, 0.0, 2}; return true; }
	return false;
}

// Growth schedule: relative growth at, growing-layer thickness H (and Hcp, zoom: visualisation only).
static void schedule(const Preset& p, double t, double& at, double& H, double& Hcp, double& zoom) {
	if (p.kind == 0) { // "Folding simulation" block of the original (commented there)
		at = 1.85*t;
		H = 0.084 - 0.042*t;
		Hcp = 0.042;
		zoom = 2.2 - 1.2*t;
	} else if (p.kind == 1) { // active block of the original
		if (t < 0.0) {
			at = 1.85 + 7.4*t;
			H = 0.041 + 0.01*t;
			Hcp = H;
		}
		if (t >= 0.0) {
			at = 1.85 - 1.85*t;
			H = 0.042 + 0.042*t;
			Hcp = 0.042;
			zoom = 1.0 + 1.2*t;
		}
	} else { // BrainGrowth simulation.py: growthRate(1.829, t), cortex_thickness = 0.042 + 0.01*t
		at = 1.829*t;
		H = 0.042 + 0.01*t;
		Hcp = H;
	}
}

static void usage() {
	cerr << "usage: tallinen_ref --mesh FILE --out DIR [options]\n"
	        "  --preset NAME        tallinen2016 (default) | tallinen2016-unfold | braingrowth\n"
	        "  --normalize MODE     none (default) | tallinen   (transform of the original's geom_norm.mesh block)\n"
	        "  --a VALUE            mesh spacing parameter a (default 0.01)\n"
	        "  --max-steps N        stop after N steps (default: run until t >= 1)\n"
	        "  --dump-every N       dump node fields every N steps (default 0 = only the steps in --dump-steps)\n"
	        "  --dump-steps LIST    comma-separated extra steps to dump (default 0)\n"
	        "  --dump-final         also dump the last step\n"
	        "  --dump-tets          also dump per-tet G (9) and J at dumped steps\n"
	        "  --log-every N        scalars.csv period (default 500, as di in the original)\n"
	        "  --midplane MODE      original (default, as in Brains.cpp) | off | restoring\n"
	        "                       (the original penalty has the wrong sign: it pushes a node that crossed\n"
	        "                        y = mpy further away; restoring flips that sign)\n"
	        "  --write-mesh FILE    write the (normalised) mesh in Netgen .mesh format and exit\n";
}

int main(int argc, char* argv[]) {

	string meshPath, outDir, presetName = "tallinen2016", normalize = "none", writeMesh, midplane = "original";
	double a_param = 0.01;
	long maxSteps = -1;
	int dumpEvery = 0, logEvery = 500;
	vector<long> dumpSteps = {0};
	bool dumpFinal = false, dumpTets = false;
	for (int k = 1; k < argc; k++) {
		string s = argv[k];
		auto next = [&]() -> string { if (k + 1 >= argc) { usage(); exit(2); } return argv[++k]; };
		if (s == "--mesh") meshPath = next();
		else if (s == "--out") outDir = next();
		else if (s == "--preset") presetName = next();
		else if (s == "--normalize") normalize = next();
		else if (s == "--a") a_param = atof(next().c_str());
		else if (s == "--max-steps") maxSteps = atol(next().c_str());
		else if (s == "--dump-every") dumpEvery = atoi(next().c_str());
		else if (s == "--dump-steps") { dumpSteps.clear(); stringstream ss(next()); string x; while (getline(ss, x, ',')) if (!x.empty()) dumpSteps.push_back(atol(x.c_str())); }
		else if (s == "--dump-final") dumpFinal = true;
		else if (s == "--dump-tets") dumpTets = true;
		else if (s == "--log-every") logEvery = atoi(next().c_str());
		else if (s == "--write-mesh") writeMesh = next();
		else if (s == "--midplane") midplane = next();
		else { usage(); return 2; }
	}
	Preset preset;
	if (meshPath.empty() || (outDir.empty() && writeMesh.empty()) || !get_preset(presetName, preset) || (normalize != "none" && normalize != "tallinen") || (midplane != "original" && midplane != "off" && midplane != "restoring")) { usage(); return 2; }

	ifstream filu(meshPath);
	if (!filu) { cerr << "cannot open " << meshPath << endl; return 1; }

	// Read nodes
	int nn; filu >> nn; // Number of nodes
	Vector* Ut = new Vector[nn](); // Positions if deformed state
	Vector* Ut0 = new Vector[nn](); // Positions in reference state
	for (int i = 0; i < nn; i++) {
		double x, y, z;
		filu >> x >> y >> z;
		Ut0[i] = Vector(x, y, z);
		Ut[i] = Ut0[i];
	}
	// Read elements
	int ne; filu >> ne; // Number of tetrahedrons
	Tetra* tets = new Tetra[ne]();
	for (int i = 0; i < ne; i++) {
		int dom, b, c, d, e;
		filu >> dom >> b >> c >> d >> e;
		tets[i] = Tetra(b-1, c-1, e-1, d-1); // Note the switch of handedness - the code uses right handed tets
	}
	// Read faces
	int nf; filu >> nf; // Number of surface triangles
	vector<Face> faces;
	for (int i = 0; i < nf; i++) {
		int bc, b, c, d;
		filu >> bc >> b >> c >> d;
		faces.push_back(Face(b-1, c-1, d-1));
	}
	if (!filu) { cerr << "error while reading " << meshPath << endl; return 1; }
	filu.close();

	// Optional normalisation: same transform as the commented "Write mesh" block of the original
	// (x' = -(y-cog.y)/maxd, y' = (x-cog.x)/maxd, z' = -(z-cog.z)/maxd), identical to BrainGrowth's
	// netgen_to_array + normalise_coord for a whole brain.
	double maxd = 1.0;
	Vector cogN;
	if (normalize == "tallinen") {
		double maxx = -1e9, minx = 1e9, maxy = -1e9, miny = 1e9, maxz = -1e9, minz = 1e9;
		for (int i = 0; i < nn; i++) {
			maxx = max(maxx, Ut[i].x); minx = min(minx, Ut[i].x);
			maxy = max(maxy, Ut[i].y); miny = min(miny, Ut[i].y);
			maxz = max(maxz, Ut[i].z); minz = min(minz, Ut[i].z);
			cogN += Ut[i];
		}
		cogN /= nn;
		maxd = max(max(max(abs(maxx-cogN.x), abs(minx-cogN.x)), max(abs(maxy-cogN.y), abs(miny-cogN.y))), max(abs(maxz-cogN.z), abs(minz-cogN.z)));
		for (int i = 0; i < nn; i++) {
			Vector u = Ut[i];
			Ut0[i] = Vector(-(u.y - cogN.y)/maxd, (u.x - cogN.x)/maxd, -(u.z - cogN.z)/maxd);
			Ut[i] = Ut0[i];
		}
	}

	// Orientation check. The reader expects right-handed tets after its (1,2,4,3) switch; the
	// normalisation transform above is a reflection, so it flips every tet. Re-orient elements
	// consistently (swap n3/n4) and surface triangles (swap n2/n3) so that tet volumes are positive
	// and triangle normals point outwards (the contact code uses the triangle normal sign).
	int nneg = 0;
	for (int i = 0; i < ne; i++) {
		Matrix A(Ut0[tets[i].n2] - Ut0[tets[i].n1], Ut0[tets[i].n3] - Ut0[tets[i].n1], Ut0[tets[i].n4] - Ut0[tets[i].n1]);
		if (A.det() < 0.0) nneg++;
	}
	bool tetsFlipped = false, facesFlipped = false;
	if (nneg > 0 && nneg < ne) cerr << "WARNING: " << nneg << " of " << ne << " tets have negative volume (inconsistent orientation), left as is" << endl;
	if (nneg == ne) {
		for (int i = 0; i < ne; i++) swap(tets[i].n3, tets[i].n4);
		tetsFlipped = true;
		cerr << "all tets were left-handed: swapped n3/n4" << endl;
	}
	double encl = 0.0;
	for (int i = 0; i < nf; i++) encl += Ut0[faces[i].n1].dot(Ut0[faces[i].n2].cross(Ut0[faces[i].n3]))/6.0;
	if (encl < 0.0) {
		for (int i = 0; i < nf; i++) swap(faces[i].n2, faces[i].n3);
		facesFlipped = true;
		cerr << "surface triangles were oriented inwards: swapped n2/n3" << endl;
	}

	if (!writeMesh.empty()) {
		// Netgen format, tets written back in the file's own order (inverse of the handedness switch)
		ofstream ofilu(writeMesh);
		ofilu << setprecision(17);
		ofilu << nn << "\n";
		for (int i = 0; i < nn; i++) ofilu << Ut0[i].x << " " << Ut0[i].y << " " << Ut0[i].z << "\n";
		ofilu << ne << "\n";
		for (int i = 0; i < ne; i++) ofilu << "1 " << tets[i].n1+1 << " " << tets[i].n2+1 << " " << tets[i].n4+1 << " " << tets[i].n3+1 << "\n";
		ofilu << nf << "\n";
		for (int i = 0; i < nf; i++) ofilu << "1 " << faces[i].n1+1 << " " << faces[i].n2+1 << " " << faces[i].n3+1 << "\n";
		return 0;
	}

	// Determine surface nodes and index maps
	int nsn = 0; // Number of nodes at the surface
	vector<int> SNb(nn); // Nodal index map from full mesh to surface
	for (int i = 0; i < nn; i++) SNb[i] = 0;
	for (int i = 0; i < nf; i++) { SNb[faces[i].n1] = 1; SNb[faces[i].n2] = 1; SNb[faces[i].n3] = 1; }
	for (int i = 0; i < nn; i++) if (SNb[i] == 1) nsn++;
	vector<int> SN(nsn); // Nodal index map from surface to full mesh
	int p = 0; // Iterator
	for (int i = 0; i < nn; i++) if (SNb[i] == 1) { SN[p] = i; SNb[i] = p; p++; }

	// PARAMETERS
	double H = 0.042; // Thickness of growing layer
	double Hcp = 0.042; // Cortical plate thickness for visualization
	const double mug = 1.0; // Shear modulus of gray matter
	const double muw = 1.167; // Shear modulus of white matter
	const double K = 5.0; // Bulk modulus
	const double a = a_param; // Mesh spacing - set manually based on the average spacing in the mesh
	const double rho = preset.rho; // Mass density - adjust to run the simulation faster or slower
	double gamma = 0.5; // Damping coefficent

	double at = 0.0; // Relative growth

	const double bw = 3.2; // Width of a bounding box, centered at origin, that encloses the whole geometry even after growth
	const double mw = 8.0*a; // Width of a cell in the linked cell algorithm for proximity detection
	const double hs = 0.6*a; // Thickness of proximity skin
	const double hc = 0.2*a; // Thickness of repulsive skin
	const double kc = 10.0*K; // Contact stiffness

	const double dt = preset.dt_factor*sqrt(rho*a*a/K); // Time step
	const double eps = 0.1;
	const double k = 0.0;

	const double mpy = -0.004; // Midplane position
	// END OF PARAMETERS

	double t = preset.t0; // Current time
	int step = 0; // Current timestep
	Vector* Vt = new Vector[nn](); // Velocities
	Vector* Ft = new Vector[nn](); // Forces
	double* Vn0 = new double[nn]; // Nodal volumes in reference state
	double* Vn = new double[nn]; // Deformed nodal volumes
	Matrix I; // Unit matrix
	double Ue; // Elastic energy
	double zoom = 1.0; // Zoom variable for visualization

	vector<int>* NNLt = new vector<int>[nsn]; // Triangle-proximity lists for surface nodes
	double maxDist; // Maximum displacement since the last update of the proximity list
	Vector* Utold = new Vector[nsn]; // Stores positions when proximity list is updated
	int tp; // Loop iterator
	double ub, vb, wb; // Barycentric coordinates

	// Determine nearest surface nodes to nodes, distances to surface, and surface normals - these are needed to set up the growth of the gray matter
	vector<int> csn(nn);	// Nearest surface nodes
	vector<double> d2s(nn);	// Distances to nearest surface nodes
	dist2surf(Ut, SN.data(), nn, nsn, csn.data(), d2s.data());
	Vector* N0 = new Vector[nsn]; // Normals in reference state
	Vector Ntmp;
	for (int i = 0; i < nf; i++) { // Find normals
		Ntmp = (Ut0[faces[i].n2] - Ut0[faces[i].n1]).cross(Ut0[faces[i].n3] - Ut0[faces[i].n1]);
		N0[SNb[faces[i].n1]] += Ntmp;
		N0[SNb[faces[i].n2]] += Ntmp;
		N0[SNb[faces[i].n3]] += Ntmp;
	}
	for (int i = 0; i < nsn; i++) N0[i].normalize();

	// Mark non-growing areas
	vector<double> gr(nn);
	for (int i = 0; i < nn; i++) {
		Vector qp = Ut0[i];
		double rqp = Vector((qp.x+0.1)*0.714, qp.y, qp.z-0.05).length();
		if ( rqp < 0.6 ) gr[i] = max(1.0 - 10.0*(0.6-rqp), 0.0);	// Ellipsoid
		else gr[i] = 1.0;
	}

	// Check minimum and maximum edge lengths at the surface
	double mine = 1e9, maxe = 0.0, ave = 0.0;
	for (int i = 0; i < nf; i++) {
		mine = min((Ut[faces[i].n2] - Ut[faces[i].n1]).length(), mine);
		mine = min((Ut[faces[i].n3] - Ut[faces[i].n1]).length(), mine);
		mine = min((Ut[faces[i].n3] - Ut[faces[i].n2]).length(), mine);
		maxe = max((Ut[faces[i].n2] - Ut[faces[i].n1]).length(), maxe);
		maxe = max((Ut[faces[i].n3] - Ut[faces[i].n1]).length(), maxe);
		maxe = max((Ut[faces[i].n3] - Ut[faces[i].n2]).length(), maxe);
		ave += (Ut[faces[i].n3] - Ut[faces[i].n2]).length() + (Ut[faces[i].n3] - Ut[faces[i].n1]).length() + (Ut[faces[i].n2] - Ut[faces[i].n1]).length();
	}
	ave /= 3.0*nf;
	cerr << "nn " << nn << " ne " << ne << " nf " << nf << " nsn " << nsn << endl;
	cerr << "mine " << mine << " ave " << ave << " maxe " << maxe << " a " << a << " dt " << dt << endl;
	if (normalize == "none" && (maxe > 1.0)) cerr << "WARNING: coordinates do not look normalised (bounding box bw = 3.2); consider --normalize tallinen" << endl;

	// Static outputs
	mkdirs(outDir + "/static");
	{
		string sd = outDir + "/static/";
		write_vec3(sd + "X0.npy", Ut0, nn);
		vector<int> tbuf(4 * (size_t)ne);
		for (int i = 0; i < ne; i++) { tbuf[4*i] = tets[i].n1; tbuf[4*i+1] = tets[i].n2; tbuf[4*i+2] = tets[i].n3; tbuf[4*i+3] = tets[i].n4; }
		write_i32(sd + "tets.npy", tbuf.data(), {(size_t)ne, 4});
		vector<int> fbuf(3 * (size_t)nf);
		for (int i = 0; i < nf; i++) { fbuf[3*i] = faces[i].n1; fbuf[3*i+1] = faces[i].n2; fbuf[3*i+2] = faces[i].n3; }
		write_i32(sd + "faces.npy", fbuf.data(), {(size_t)nf, 3});
		write_i32(sd + "surface_nodes.npy", SN.data(), {(size_t)nsn});
		write_i32(sd + "nearest_surface_node.npy", csn.data(), {(size_t)nn});
		write_f64(sd + "dist_to_surface.npy", d2s.data(), nn);
		write_vec3(sd + "surface_normals0.npy", N0, nsn);
		write_f64(sd + "gr.npy", gr.data(), nn);
	}

	// Manifest
	{
		ofstream m(outDir + "/manifest.json");
		m << setprecision(17);
		m << "{\n";
		m << "  \"generator\": \"tallinen_ref (instrumented copy of Tallinen 2016 Brains.cpp)\",\n";
		m << "  \"mesh\": \"" << meshPath << "\",\n";
		m << "  \"preset\": \"" << preset.name << "\",\n";
		m << "  \"normalize\": \"" << normalize << "\",\n";
		m << "  \"midplane\": \"" << midplane << "\",\n";
		m << "  \"normalization\": {\"maxd\": " << maxd << ", \"cog\": [" << cogN.x << ", " << cogN.y << ", " << cogN.z << "]},\n";
		m << "  \"orientation\": {\"negative_tets_read\": " << nneg << ", \"tets_flipped\": " << (tetsFlipped ? "true" : "false") << ", \"faces_flipped\": " << (facesFlipped ? "true" : "false") << "},\n";
		m << "  \"counts\": {\"nodes\": " << nn << ", \"tets\": " << ne << ", \"faces\": " << nf << ", \"surface_nodes\": " << nsn << "},\n";
		m << "  \"params\": {\"mug\": " << mug << ", \"muw\": " << muw << ", \"K\": " << K << ", \"a\": " << a << ", \"rho\": " << rho
		  << ", \"gamma\": " << gamma << ", \"dt\": " << dt << ", \"dt_factor\": " << preset.dt_factor << ", \"t0\": " << preset.t0
		  << ", \"eps\": " << eps << ", \"k\": " << k << ", \"bw\": " << bw << ", \"mw\": " << mw << ", \"hs\": " << hs << ", \"hc\": " << hc
		  << ", \"kc\": " << kc << ", \"mpy\": " << mpy << "},\n";
		m << "  \"surface_edge_length\": {\"min\": " << mine << ", \"mean\": " << ave << ", \"max\": " << maxe << "},\n";
		m << "  \"conventions\": {\"index_base\": 0, \"tets\": \"after the handedness switch of the reader (file order 1,2,4,3)\", "
		     "\"dump_point\": \"start of step: x and v before integration; f_* accumulated during the step\", "
		     "\"f_contact\": \"Ft after contact\", \"f_after_elastic\": \"Ft after contact + elasticity\", \"f_total\": \"Ft after contact + elasticity + midplane (integrated force before damping)\"}\n";
		m << "}\n";
	}

	ofstream dat(outDir + "/scalars.csv");
	dat << setprecision(17);
	dat << "step,t,at,H,Uk,Ue,Area,Volume,width_x,width_y,ninv\n";

	auto isDump = [&](int s) {
		if (dumpEvery > 0 && s % dumpEvery == 0) return true;
		for (long d : dumpSteps) if (d == s) return true;
		return false;
	};
	auto stopNext = [&](int s, double tt) { return !(tt < 1.0) || (maxSteps >= 0 && s >= maxSteps); };

	while (t < 1.0 && (maxSteps < 0 || step < maxSteps)) { // Main loop

		schedule(preset, t, at, H, Hcp, zoom);

		bool dump = isDump(step) || (dumpFinal && stopNext(step + 1, t + dt));
		string sd;
		if (dump) { sd = outDir + "/" + fmt_step(step) + "/"; mkdirs(sd); write_vec3(sd + "x.npy", Ut, nn); write_vec3(sd + "v.npy", Vt, nn); }

		// Contact processing
		maxDist = 0.0;
		for (int i = 0; i < nsn; i++ ) { // Find the maximum displacement since the last update of the proximity list
			maxDist = max(maxDist, (Ut[SN[i]] - Utold[i]).length());
		}
		bool rebuilt = false;
		if (maxDist > 0.5*(hs-hc)) { // Update proximity list
			createNNLtriangle(NNLt, Ut, faces, SN.data(), nsn, nf, hs, bw, mw);
			for (int i = 0; i < nsn; i++) Utold[i] = Ut[SN[i]];
			rebuilt = true;
		}
		for (int i = 0; i < nsn; i++) { // Loop through surface points
			for (tp = 0; tp < (int)NNLt[i].size(); tp++) { // Loop trough triangle-proximity list
				int pt = SN[i];
				int tri = NNLt[i][tp];
				Vector cc = closestPointTriangle(Ut[pt], Ut[faces[tri].n1], Ut[faces[tri].n2], Ut[faces[tri].n3], ub, vb, wb) - Ut[pt];
				double rc = cc.length();
				if (rc < hc && gr[pt] + gr[faces[tri].n1] > 0.0) { // Calculate contact force if within the contact range
					cc.normalize();
					Vector Ntri = (Ut[faces[tri].n2] - Ut[faces[tri].n1]).cross(Ut[faces[tri].n3] - Ut[faces[tri].n1]);
					Ntri.normalize();
					Vector fn = cc*(rc-hc)/hc*kc*a*a;
					if (fn.dot(Ntri) < 0.0) fn -= Ntri*fn.dot(Ntri)*2.0;
					Ft[faces[tri].n1] -= fn*ub;
					Ft[faces[tri].n2] -= fn*vb;
					Ft[faces[tri].n3] -= fn*wb;
					Ft[pt] += fn;
				}
			}
		}
		if (dump) {
			write_vec3(sd + "f_contact.npy", Ft, nn);
			size_t npairs = 0; for (int i = 0; i < nsn; i++) npairs += NNLt[i].size();
			ofstream ci(sd + "contact.json");
			ci << "{\"proximity_list_rebuilt\": " << (rebuilt ? "true" : "false") << ", \"proximity_pairs\": " << npairs << "}\n";
		}

		// Nodal volumes for nodal pressure calculation
		for (int i = 0; i < nn; i++) { Vn0[i] = 0.0; Vn[i] = 0.0; }
		for (int i = 0; i < ne; i++) {
			int n1 = tets[i].n1;
			int n2 = tets[i].n2;
			int n3 = tets[i].n3;
			int n4 = tets[i].n4;

			// Undeformed
			Vector xr1 = Ut0[n2] - Ut0[n1];
			Vector xr2 = Ut0[n3] - Ut0[n1];
			Vector xr3 = Ut0[n4] - Ut0[n1];
			Matrix Ar = Matrix(xr1, xr2, xr3);
			Ar = tets[i].G.prod(Ar);

			double vol0 = Ar.det()/6.0;
			Vn0[n1] += vol0/4.0;
			Vn0[n2] += vol0/4.0;
			Vn0[n3] += vol0/4.0;
			Vn0[n4] += vol0/4.0;

			// Deformed
			Vector x1 = Ut[n2] - Ut[n1];
			Vector x2 = Ut[n3] - Ut[n1];
			Vector x3 = Ut[n4] - Ut[n1];
			Matrix A = Matrix(x1, x2, x3);
			double vol = A.det()/6.0;
			Vn[n1] += vol/4.0;
			Vn[n2] += vol/4.0;
			Vn[n3] += vol/4.0;
			Vn[n4] += vol/4.0;
		}

		vector<double> tetG, tetJ;
		if (dump && dumpTets) { tetG.resize(9 * (size_t)ne); tetJ.resize(ne); }

		// Deformations
		Ue = 0.0; int ninv = 0;
		for (int i = 0; i < ne; i++) {

			// Nodal indices
			int n1 = tets[i].n1;
			int n2 = tets[i].n2;
			int n3 = tets[i].n3;
			int n4 = tets[i].n4;

			// Gray and white matter indicators
			double gm = 1.0/(1.0 + exp(10.0*(0.25*(d2s[n1]+d2s[n2]+d2s[n3]+d2s[n4])/H - 1.0))) * 0.25*(gr[n1]+gr[n2]+gr[n3]+gr[n4]);
			double wm = 1.0 - gm;
			double mu = muw*wm + mug*gm;

			// Basis vector of reference state
			Vector xr1 = Ut0[n2] - Ut0[n1];
			Vector xr2 = Ut0[n3] - Ut0[n1];
			Vector xr3 = Ut0[n4] - Ut0[n1];
			Matrix Ar = Matrix(xr1, xr2, xr3); // Reference state
			Ar = tets[i].G.prod(Ar); // Apply growth to reference state

			// Deformed basis vectors
			Vector x1 = Ut[n2] - Ut[n1];
			Vector x2 = Ut[n3] - Ut[n1];
			Vector x3 = Ut[n4] - Ut[n1];

			// Undeformed normals
			xr1 = Vector(Ar.a, Ar.d, Ar.g);
			xr2 = Vector(Ar.b, Ar.e, Ar.h);
			xr3 = Vector(Ar.c, Ar.f, Ar.i);
			Vector N1 = xr3.cross(xr1);
			Vector N2 = xr2.cross(xr3);
			Vector N3 = xr1.cross(xr2);
			Vector N4 = (xr2 - xr3).cross(xr1 - xr3);

			Matrix A = Matrix(x1, x2, x3); // Deformed state
			double vol = A.det()/6.0; // Deformed volume
			Matrix F = A.prod(Ar.inv()); // Deformation gradient
			Matrix B = F.prod(F.trans()); // Left Cauchy-Green strain tensor
			double J = F.det(); // Relative volume change
			double J1 = Vn[n1]/Vn0[n1];
			double J2 = Vn[n2]/Vn0[n2];
			double J3 = Vn[n3]/Vn0[n3];
			double J4 = Vn[n4]/Vn0[n4];
			double Ja = (J1 + J2 + J3 + J4)/4.0; // Averaged nodal volume change

			if (dump && dumpTets) {
				Matrix& G = tets[i].G;
				double* g = &tetG[9 * (size_t)i];
				g[0] = G.a; g[1] = G.b; g[2] = G.c; g[3] = G.d; g[4] = G.e; g[5] = G.f; g[6] = G.g; g[7] = G.h; g[8] = G.i;
				tetJ[i] = J;
			}

			double powJ23, W;
			Matrix P;
			if (B.EV().z >= eps*eps && J > 0.0) { // No need for SVD

				powJ23 = pow(J, 2.0/3.0);
				Matrix T = (B - I*B.trace()/3.0)*mu/(J*powJ23) + I*K*(Ja-1.0);
				P = T.prod(F.trans().inv())*J;
				W = 0.5*mu*(B.trace()/powJ23 - 3.0) + 0.5*K*( (J1-1.0)*(J1-1.0) + (J2-1.0)*(J2-1.0) + (J3-1.0)*(J3-1.0) + (J4-1.0)*(J4-1.0) )*0.25;
			} else { // Needs SVD

				Matrix C = F.trans().prod(F);
				Matrix V;
				double eva[3];
				Eigensystem(C, V, eva);

				double l1 = sqrt(eva[0]);
				double l2 = sqrt(eva[1]);
				double l3 = sqrt(eva[2]);

				if (V.det() < 0.0) { V.a = -V.a; V.d = -V.d; V.g = -V.g; }

				Matrix Fdi;
				if (l1 >= 1e-25) Fdi.a = 1.0/l1;
				Fdi.e = 1.0/l2;
				Fdi.i = 1.0/l3;

				Matrix U = F.prod(V.prod(Fdi));

				if (l1 < 1e-25) {
					U.a = U.e*U.i - U.h*U.f;
					U.d = U.h*U.c - U.b*U.i;
					U.g = U.b*U.f - U.e*U.c;
				}
				if (F.det() < 0.0) {
					ninv++;
					l1 = -l1;
					U.a = -U.a; U.d = -U.d; U.g = -U.g;
				}

				Matrix Pd;
				double pow23 = pow(eps*l2*l3, 2.0/3.0);
				Pd.a = mu/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k*(l1-eps) + K*(Ja-1.0)*l2*l3;
				Pd.e = mu/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + K*(Ja-1.0)*l1*l3;
				Pd.i = mu/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + K*(Ja-1.0)*l1*l2;
				W = 0.5*mu*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k*(l1-eps)*(l1-eps) + 0.5*K*((J1-1.0)*(J1-1.0) + (J2-1.0)*(J2-1.0) + (J3-1.0)*(J3-1.0) + (J4-1.0)*(J4-1.0))/4.0;

				P = U.prod(Pd.prod(V.trans()));
			}

			if (J*J > 1e-50) Ue += W*vol/J; // Increment total elastic energy

			// Apply forces to nodes
			Ft[n1] += P.prod(N1 + N2 + N3)/6.0;
			Ft[n2] += P.prod(N1 + N3 + N4)/6.0;
			Ft[n3] += P.prod(N2 + N3 + N4)/6.0;
			Ft[n4] += P.prod(N1 + N2 + N4)/6.0;

			// Growth
			Vector Ns = (N0[csn[n1]] + N0[csn[n2]] + N0[csn[n3]] + N0[csn[n4]]); // Surface normal
			Ns.normalize();
			tets[i].G = I + (I - Matrix(Ns.x*Ns.x, Ns.x*Ns.y, Ns.x*Ns.z, Ns.x*Ns.y, Ns.y*Ns.y, Ns.y*Ns.z, Ns.x*Ns.z, Ns.y*Ns.z, Ns.z*Ns.z))*gm*at;
		}
		if (dump) {
			write_vec3(sd + "f_after_elastic.npy", Ft, nn);
			write_f64(sd + "Vn.npy", Vn, nn);
			write_f64(sd + "Vn0.npy", Vn0, nn);
			if (dumpTets) {
				write_npy(sd + "G_used.npy", tetG.data(), "<f8", 8, {(size_t)ne, 3, 3});
				write_f64(sd + "J.npy", tetJ.data(), ne);
			}
		}

		// Midplane
		const double mpSign = (midplane == "restoring") ? -1.0 : 1.0;
		if (midplane != "off") for (int i = 0; i < nsn; i++) {
			int pt = SN[i];
			if ( Ut0[pt].y < mpy - 0.5*a && Ut[pt].y > mpy ) {
				Ft[pt].y -= mpSign*(mpy - Ut[pt].y)/hc*a*a*K;
			}
			if ( Ut0[pt].y > mpy + 0.5*a && Ut[pt].y < mpy ) {
				Ft[pt].y -= mpSign*(mpy - Ut[pt].y)/hc*a*a*K;
			}
		}
		if (dump) {
			write_vec3(sd + "f_total.npy", Ft, nn);
			ofstream si(sd + "step.json");
			si << setprecision(17) << "{\"step\": " << step << ", \"t\": " << t << ", \"at\": " << at << ", \"H\": " << H << ", \"Ue\": " << Ue << ", \"ninv\": " << ninv << "}\n";
		}

		// Output
		if (logEvery > 0 && step%logEvery == 0) {
			double Uk = 0.0; // Kinetic energy
			for (int i = 0; i < nn; i++) Uk += 0.5*(Vn[i]*rho)*(Vt[i].dot(Vt[i]));
			double Area = 0.0; // Surface area
			for (int i = 0; i < nf; i++) {
				Vector N = (Ut[faces[i].n2]-Ut[faces[i].n1]).cross(Ut[faces[i].n3]-Ut[faces[i].n1]);
				Area += 0.5*N.length()*(gr[faces[i].n1] + gr[faces[i].n2] + gr[faces[i].n3])/3.0; // Only growing area
			}
			double Volume = 0.0; // Volume
			for (int i = 0; i < nn; i++) Volume += Vn[i];
			double ymin = 1.0, ymax = -1.0, xmin = 1.0, xmax = -1.0;
			for (int i = 0; i < nsn; i++) {
				xmin = min(xmin, Ut[SN[i]].x);
				xmax = max(xmax, Ut[SN[i]].x);
				ymin = min(ymin, Ut[SN[i]].y);
				ymax = max(ymax, Ut[SN[i]].y);
			}
			dat << step << "," << t << "," << at << "," << H << "," << Uk << "," << Ue << "," << Area << "," << Volume << "," << xmax-xmin << "," << ymax-ymin << "," << ninv << "\n";
			dat.flush();
			cerr << "step " << step << " t " << t << " Ue " << Ue << " Area " << Area << " ninv " << ninv << endl;
		}

		// Newton dynamics
		for (int i = 0; i < nn; i++) {
			Ft[i] -= Vt[i]*gamma*Vn0[i];
			Vt[i] += Ft[i]/(Vn0[i]*rho)*dt;
			Ut[i] += Vt[i]*dt;
			Ft[i].clear();
		}
		t += dt;
		step++;
	}
	dat.close();
	{
		mkdirs(outDir + "/final");
		write_vec3(outDir + "/final/x.npy", Ut, nn);
		write_vec3(outDir + "/final/v.npy", Vt, nn);
		ofstream si(outDir + "/final/step.json");
		si << setprecision(17) << "{\"step\": " << step << ", \"t\": " << t << "}\n";
	}
	return 0;
}

void Eigensystem(Matrix A, Matrix& V, double d[3]) {

	double A_[3][3];
	double V_[3][3];

	A_[0][0] = A.a; A_[0][1] = A.b; A_[0][2] = A.c;
	A_[1][0] = A.d; A_[1][1] = A.e; A_[1][2] = A.f;
	A_[2][0] = A.g; A_[2][1] = A.h; A_[2][2] = A.i;

	eigen_decomposition(A_, V_, d);

	V.a = V_[0][0]; V.b = V_[0][1]; V.c = V_[0][2];
	V.d = V_[1][0]; V.e = V_[1][1]; V.f = V_[1][2];
	V.g = V_[2][0]; V.h = V_[2][1]; V.i = V_[2][2];
}

// Generates point-triangle proximity lists using the linked cell algorithm
void createNNLtriangle(vector<int>* NNLt, Vector* Ut, vector<Face>& faces, int* SN, int nsn, int nf, double hs, double bw, double mw) {

	int mx = max(1, (int)(bw/mw));
	vector<int> head(mx*mx*mx, -1);
	vector<int> list(nf);
	double ub, vb, wb;

	for (int i = 0; i < nf; i++) { // Divide triangles into cells
		Vector cog = (Ut[faces[i].n1] + Ut[faces[i].n2] + Ut[faces[i].n3])/3.0;
		int xa = (int)((cog.x + 0.5*bw)/bw*mx);
		int ya = (int)((cog.y + 0.5*bw)/bw*mx);
		int za = (int)((cog.z + 0.5*bw)/bw*mx);
		if (xa < 0 || ya < 0 || za < 0 || xa >= mx || ya >= mx || za >= mx) {
			cerr << "FATAL: triangle " << i << " outside the contact bounding box (bw = " << bw << "); the original code would write out of bounds" << endl;
			exit(3);
		}
		list[i]=head[mx*mx*za + mx*ya + xa];
		head[mx*mx*za + mx*ya + xa] = i;
	}
	for (int i = 0; i < nsn; i++) { // Search cells around each point and build proximity list
		int pt = SN[i];
		NNLt[i].clear();
		int xa = (int)((Ut[pt].x + 0.5*bw)/bw*mx);
		int ya = (int)((Ut[pt].y + 0.5*bw)/bw*mx);
		int za = (int)((Ut[pt].z + 0.5*bw)/bw*mx);
		for (int xi = max(0, xa-1); xi <= min(mx-1, xa+1); xi++)
		for (int yi = max(0, ya-1); yi <= min(mx-1, ya+1); yi++)
		for (int zi = max(0, za-1); zi <= min(mx-1, za+1); zi++) {
			int tri = head[mx*mx*zi + mx*yi + xi];
			while (tri != -1) {
				if ( pt != faces[tri].n1 && pt != faces[tri].n2 && pt != faces[tri].n3 ) {
					if ( (closestPointTriangle(Ut[pt], Ut[faces[tri].n1], Ut[faces[tri].n2], Ut[faces[tri].n3], ub, vb, wb) - Ut[pt]).length() < hs) {
						NNLt[i].push_back(tri);
					}
				}
				tri = list[tri];
			}
		}
	}
}
// Returns the closest point of triangle abc to point p
Vector closestPointTriangle(Vector& p, Vector& a, Vector& b, Vector& c, double& u, double& v, double& w) {

	Vector ab = b - a;
	Vector ac = c - a;
	Vector ap = p - a;
	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);
	if (d1 <= 0.0 && d2 <= 0.0) {
		u = 1.0;
		v = 0.0;
		w = 0.0;
		return a;
	}
	Vector bp = p - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3) {
		u = 0.0;
		v = 1.0;
		w = 0.0;
		return b;
	}
	double vc = d1*d4 - d3*d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
		v = d1 / (d1 - d3);
		u = 1.0 - v;
		w = 0.0;
		return a + ab * v;
	}
	Vector cp = p - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6) {
		u = 0.0;
		v = 0.0;
		w = 1.0;
		return c;
	}
	double vb = d5*d2 - d1*d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
		w = d2 / (d2 - d6);
		u = 1.0 - w;
		v = 0.0;
		return a + ac * w;
	}
	double va = d3*d6 - d5*d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
		w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		u = 0.0;
		v = 1.0 - w;
		return b + (c - b) * w;
	}
	double denom = 1.0 / (va + vb + vc);
	v = vb * denom;
	w = vc * denom;
	u = 1.0 - v - w;
	return a + ab * v + ac * w;
}

// Finds the nearest surface nodes (csn) to nodes and distances to them (d2s)
void dist2surf(Vector* Ut, int* SN, int nn, int nsn, int* csn, double* d2s) {
	int p = 0, j;
	for (int i = 0; i < nn; i++) {
		double d2min = 1e9;
		for (j = 0; j < nsn; j++) {
			double d2 = (Ut[SN[j]] - Ut[i]).dot(Ut[SN[j]] - Ut[i]);
			if (d2 < d2min) {
				d2min = d2;
				p = j;
			}
		}
		csn[i] = p;
		d2s[i] = sqrt(d2min);
	}
}
