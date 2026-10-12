//
//	Compile example with gnu: g++ -Ofast -fopenmp Eig3.cpp Brains.cpp -o Brains
//	vema.h and eig3.h has to be in the same directory
//
#include <omp.h>
#include "vema.h"
#include "eig3.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
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
void writeAVIZO(Vector*, int, Tetra*, int, double*, double*, double, double, int);
void writePov(Vector*, vector<Face>&, int*, int*, int, int, double);

int main(int argc, char* argv[]) {

	char inpa[10], inpb[10], inpc[10], inpd[10], inpe[10];
	ifstream filu;
	filu.open(getenv("MESH")); // Should be in Netgen's .mesh format
	
	// Read nodes
	filu >> inpa;
	int nn = atoi(inpa); // Number of nodes
	Vector* Ut = new Vector[nn](); // Positions if deformed state
	Vector* Ut0 = new Vector[nn](); // Positions in reference state
	for (int i = 0; i < nn; i++) {
		filu >> inpa >> inpb >> inpc;
		Ut0[i] = Vector(atof(inpa), atof(inpb), atof(inpc));
		Ut[i] = Ut0[i];
	}
	// Read elements
	filu >> inpa;
	int ne = atoi(inpa); // Number of tetrahedrons
	Tetra* tets = new Tetra[ne]();
	for (int i = 0; i < ne; i++) {
		filu >> inpa >> inpb >> inpc >> inpd >> inpe;
		tets[i] = Tetra(atoi(inpb)-1, atoi(inpc)-1, atoi(inpe)-1, atoi(inpd)-1); // Note the switch of handedness - the code uses right handed tets
	}
	// Read faces
	filu >> inpa;
	int nf = atoi(inpa); // Number of surface triangles
	vector<Face> faces;
	for (int i = 0; i < nf; i++) {
		filu >> inpa >> inpb >> inpc >> inpd;
		faces.push_back(Face(atoi(inpb)-1, atoi(inpc)-1, atoi(inpd)-1));
	}
	filu.close();	
	
	// Determine surface nodes and index maps
	int nsn = 0; // Number of nodes at the surface
	int SNb[nn]; // Nodal index map from full mesh to surface
	for (int i = 0; i < nn; i++) SNb[i] = 0;
	for (int i = 0; i < nf; i++) { SNb[faces[i].n1] = 1; SNb[faces[i].n2] = 1; SNb[faces[i].n3] = 1; }
	for (int i = 0; i < nn; i++) if (SNb[i] == 1) nsn++;
	int SN[nsn]; // Nodal index map from surface to full mesh
	int p = 0; // Iterator
	for (int i = 0; i < nn; i++) if (SNb[i] == 1) { SN[p] = i; SNb[i] = p; p++; }

	// PARAMETERS
	double H = 0.042; // Thickness of growing layer
	double Hcp = 0.042; // Cortical plate thickness for visualization
	const double mug = 1.0; // Shear modulus of gray matter
	const double muw = 1.167; // Shear modulus of white matter
	const double K = 5.0; // Bulk modulus
	const double a = 0.01; // Mesh spacing - set manually based on the average spacing in the mesh
	const double rho = 0.0025; // Mass density - adjust to run the simulation faster or slower
	double gamma = 0.5; // Damping coefficent
	const int di = 100; // Output data once every di steps
	
	double at; // Relative growth
	/*double alfagt = log(3.0); // Tangential growth rate of gray matter
	double alfagn = log(0.5); // Growth rate of gray matter in thickness direction
	double alfaw = log(1.0); // Isotropic growth rate of white matter
	double alfar = 0.5; // Rate of stress relaxation*/
	
	const double bw = 3.2; // Width of a bounding box, centered at origin, that encloses the whole geometry even after growth
	const double mw = 8.0*a; // Width of a cell in the linked cell algorithm for proximity detection
	const double hs = 0.6*a; // Thickness of proximity skin
	const double hc = 0.2*a; // Thickness of repulsive skin
	const double kc = 10.0*K; // Contact stiffness
			
	const double dt = 0.025*sqrt(rho*a*a/K); // Time step
	const double eps = 0.1;
	const double k = 0.0;
	
	const double mpy = -0.004; // Midplane position
	// END OF PARAMETERS
		
	double t = -0.25; // Current time
	//double t = 0.0; // Current time
	int step = 0; // Current timestep
	Vector* Vt = new Vector[nn](); // Velocities
	Vector* Ft = new Vector[nn](); // Forces
	double* Vn0 = new double[nn]; // Nodal volumes in reference state
	double* Vn = new double[nn]; // Deformed nodal volumes
	Matrix I; // Unit matrix
	double Ue; // Elastic energy
	double zoom = 1.0; // Zoom variable for visualization
	
	vector<int> NNLt[nsn]; // Triangle-proximity lists for surface nodes
	double maxDist; // Maximum displacement since the last update of the proximity list
	Vector* Utold = new Vector[nsn]; // Stores positions when proximity list is updated
	int tp; // Loop iterator
	double ub, vb, wb; // Barycentric coordinates
			
	// Determine nearest surface nodes to nodes, distances to surface, and surface normals - these are needed to set up the growth of the gray matter
	int csn[nn];	// Nearest surface nodes
	double d2s[nn];	// Distances to nearest surface nodes
	dist2surf(Ut, SN, nn, nsn, csn, d2s);
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
	double gr[nn];
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
	cout << "mine	" << mine  << "ave	" << ave << "	maxe	" << maxe << "	a	" << a << endl;
	
	// Find center of mass and dimension of the mesh
	double maxx = -1e9, minx = 1e9, maxy = -1e9, miny = 1e9, maxz = -1e9, minz = 1e9;
	Vector cog;
	for (int i = 0; i < nn; i++) {
		maxx = max(maxx, Ut[i].x); minx = min(minx, Ut[i].x);
		maxy = max(maxy, Ut[i].y); miny = min(miny, Ut[i].y);
		maxz = max(maxz, Ut[i].z); minz = min(minz, Ut[i].z);
		cog += Ut[i];
	}
	cog /= nn;
		
	double maxd = max(max(max(abs(maxx-cog.x), abs(minx-cog.x)), max(abs(maxy-cog.y), abs(miny-cog.y))), max(abs(maxz-cog.z), abs(minz-cog.z)));
	
	cout << "cog " << cog.x << " " << cog.y << " " << cog.z << endl;
	cout << minx << " " << maxx << endl;
	cout << miny << " " << maxy << endl;
	cout << minz << " " << maxz << endl;
	cout << maxd << endl;
	
	// Write mesh
	/*ofstream ofilu("geom_norm.mesh");
	ofilu << fixed;
	ofilu << nn << endl;
	for (int i = 0; i < nn; i++) ofilu << setw(11) << -(Ut[i].y - cog.y)/maxd << setw(11) << (Ut[i].x - cog.x)/maxd << setw(11) << -(Ut[i].z - cog.z)/maxd << endl;
	ofilu << ne << endl;
	for (int i = 0; i < ne; i++) ofilu << "1" << setw(11) << tets[i].n1+1 << setw(11) << tets[i].n2+1 << setw(11) << tets[i].n4+1 << setw(11) << tets[i].n3+1 << endl;
	ofilu << nf << endl;
	for (int i = 0; i < nf; i++) ofilu << "1" << setw(11) << faces[i].n1+1 << setw(11) << faces[i].n2+1 << setw(11) << faces[i].n3+1 << endl;
	ofilu.close();*/
	
	cout.precision(3);
	ofstream dat;
	dat.precision(17);
	dat.open("Brains.dat");
		
	long maxSteps = atol(getenv("MAXSTEPS")); while (t < 1.0 && step < maxSteps) { // Main loop
		
		// Unfolding simulation
		if (t < 0.0) { // Fold the "adult brain" 
			at = 1.85 + 7.4*t;
			H = 0.041 + 0.01*t;
			Hcp = H;
		}
		if (t >= 0.0) { // Unfold the brain following a reverse developmental path
			at = 1.85 - 1.85*t;
			H = 0.042 + 0.042*t;
			Hcp = 0.042;
			zoom = 1.0 + 1.2*t;
		}
		
		// Folding simulation
		/*at = 1.85*t;
		H = 0.084 - 0.042*t;
		Hcp = 0.042;
		zoom = 2.2 - 1.2*t;*/
	
		// Contact processing
		maxDist = 0.0;
		#pragma omp parallel for reduction(max:maxDist)
		for (int i = 0; i < nsn; i++ ) { // Find the maximum displacement since the last update of the proximity list
			maxDist = max(maxDist, (Ut[SN[i]] - Utold[i]).length());
		}		
		if (maxDist > 0.5*(hs-hc)) { // Update proximity list
			createNNLtriangle(NNLt, Ut, faces, SN, nsn, nf, hs, bw, mw);
			cout << step << endl;
			for (int i = 0; i < nsn; i++) Utold[i] = Ut[SN[i]];
		}
		#pragma omp parallel for private(tp)
		for (int i = 0; i < nsn; i++) { // Loop through surface points
			for (tp = 0; tp < NNLt[i].size(); tp++) { // Loop trough triangle-proximity list
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
			
		// Nodal volumes for nodal pressure calculation
		#pragma omp parallel for
		for (int i = 0; i < nn; i++) { Vn0[i] = 0.0; Vn[i] = 0.0; }
		#pragma omp parallel for
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
			
		// Deformations
		Ue = 0.0; int ninv = 0;
		#pragma omp parallel for reduction(+:Ue, ninv)
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
						
			double powJ23, W;
			Matrix P;
			if (B.EV().z >= eps*eps && J > 0.0) { // No need for SVD
			
				//powJ23 = 1.0 + 2.0/3.0*(J - 1.0) - 1.0/9.0*(J-1.0)*(J-1.0); // Approximate pow(J, 2/3)
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
			
			// Mandel stress - for modeling stress relaxation
			/*if (J > 1e-25) {	
				Matrix M = F.trans().prod(P)*J;
				M -= I*M.trace()/3.0; // Remove isotropic stress
				Matrix Lp = M*alfar;
				tets[i].G += Lp.prod(tets[i].G)*dt;
			}*/
			
			// Growth
			Vector Ns = (N0[csn[n1]] + N0[csn[n2]] + N0[csn[n3]] + N0[csn[n4]]); // Surface normal
			Ns.normalize();
			/*Matrix Lggt = (I - Matrix(Ns.x*Ns.x, Ns.x*Ns.y, Ns.x*Ns.z, Ns.x*Ns.y, Ns.y*Ns.y, Ns.y*Ns.z, Ns.x*Ns.z, Ns.y*Ns.z, Ns.z*Ns.z))*alfagt; // Tangential growth rate of gray matter
			Matrix Lggn = Matrix(Ns.x*Ns.x, Ns.x*Ns.y, Ns.x*Ns.z, Ns.x*Ns.y, Ns.y*Ns.y, Ns.y*Ns.z, Ns.x*Ns.z, Ns.y*Ns.z, Ns.z*Ns.z)*alfagn; // Growth rate of the gray matter in thickness direction
			Matrix Lgg = Lggt + Lggn; // Total tensorial growth rate gray matter
			Matrix Lgw = I*alfaw; // Isotropic growth rate of white matter
			Matrix Lg = Lgg*gm + Lgw*wm; // Total growth rate
			tets[i].G += Lg.prod(tets[i].G)*dt; // Update growth tensor*/
			
			tets[i].G = I + (I - Matrix(Ns.x*Ns.x, Ns.x*Ns.y, Ns.x*Ns.z, Ns.x*Ns.y, Ns.y*Ns.y, Ns.y*Ns.z, Ns.x*Ns.z, Ns.y*Ns.z, Ns.z*Ns.z))*gm*at;
		}
		
		// Midplane
		#pragma omp parallel for 
		for (int i = 0; i < nsn; i++) {
			int pt = SN[i];
			if ( Ut0[pt].y < mpy - 0.5*a && Ut[pt].y > mpy ) {
				Ft[pt].y -= (mpy - Ut[pt].y)/hc*a*a*K;
			}
			if ( Ut0[pt].y > mpy + 0.5*a && Ut[pt].y < mpy ) {
				Ft[pt].y -= (mpy - Ut[pt].y)/hc*a*a*K;
			}
		}
				
		// Output
		if (step%di == 0) {
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
			cout  << setw(11) << step << setw(11) << t << setw(11) << Uk << setw(11) << Ue << setw(11) << Area << setw(11) << Volume << setw(11) << xmax-xmin << setw(11) << ymax-ymin << setw(11) << ninv << endl;
			dat << setw(26) << step << setw(26) << t << setw(26) << Uk << setw(26) << Ue << setw(26) << Area << setw(26) << Volume << setw(26) << xmax-xmin << setw(26) << ymax-ymin << setw(26) << ninv << endl;
			//writePov(Ut, faces, SN, SNb, nsn, step, zoom);
		}
		//if ( t >= -dt && t < 0.0) writeAVIZO(Ut, nn, tets, ne, gr, d2s, H, Hcp, step);
		//if ( t >= 0.0 && step%(20*di) == 0 ) writeAVIZO(Ut, nn, tets, ne, gr, d2s, H, Hcp, step);
		
		// Newton dynamics
		#pragma omp parallel for 
		for (int i = 0; i < nn; i++) {
			//if (gr[i] == 0.0) Ft[i].clear();
			Ft[i] -= Vt[i]*gamma*Vn0[i];
			Vt[i] += Ft[i]/(Vn0[i]*rho)*dt;
			Ut[i] += Vt[i]*dt;
			Ft[i].clear();
		}
		t += dt;
		step++;
	}		
	dat.close();
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
	int xa, ya, za, xi, yi, zi;
	double ub, vb, wb;
	int pt, tri;
	Vector cog;
	
	for (int i = 0; i < nf; i++) { // Divide triangles into cells
		Vector cog = (Ut[faces[i].n1] + Ut[faces[i].n2] + Ut[faces[i].n3])/3.0;
		int xa = (int)((cog.x + 0.5*bw)/bw*mx);
		int ya = (int)((cog.y + 0.5*bw)/bw*mx);
		int za = (int)((cog.z + 0.5*bw)/bw*mx);
		list[i]=head[mx*mx*za + mx*ya + xa];
		head[mx*mx*za + mx*ya + xa] = i;
	}
	#pragma omp parallel for
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
	int p, j;
	#pragma omp parallel for private(p, j)
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

void writeAVIZO(Vector* Ut, int nn, Tetra* tets, int ne, double* gr, double* d2s, double H, double Hcp, int step) {

	char fname[50];
	sprintf(fname, "A%d.am", step);
	ofstream avz(fname);
	avz.setf(ios::fixed);
	avz.precision(5);

	avz << "# AmiraMesh ASCII 1.0\n";
	avz << "define Nodes " << nn << endl;
	avz << "define Tetrahedra " << ne << endl;
	avz << "Nodes { float[3] Coordinates } = @1\n";
    avz << "Tetrahedra { int[4] Nodes } = @4\n";
    avz << "Nodes { float Values } = @8\n";
    avz << "Field { float GMWM } = Linear(@8)\n";

    avz << "@1\n";
    for (int i = 0; i < nn; i++) {
		avz << Ut[i].x << " " << Ut[i].y << " " << Ut[i].z << endl;
	}
	
    avz << "@4\n";
    for (int i = 0; i < ne; i++) {
		avz << tets[i].n1 + 1 << " " << tets[i].n2 + 1 << " " << tets[i].n3 + 1 << " " << tets[i].n4 + 1 << endl;
	}

    avz << "@8\n";
    for (int i = 0; i < nn; i++) {
		double cval = 1.0/(1.0 + exp(10.0*(d2s[i]/H - 1.0)))*gr[i]; // Neocortical anlage
		cval += 1.0/(1.0 + exp(10.0*(d2s[i]/Hcp - 1.0)))*gr[i]; // Cortical plate
		avz << 3.0 - cval << endl;
	}
	
	avz.close();
}
// Writes POV-Ray source files
void writePov(Vector* Ut, vector<Face>& faces, int* SN, int* SNb, int nsn, int step, double zoom) {

	char povname[50];
	sprintf(povname, "B%d.pov", step);
	ofstream pov(povname);
	pov.setf(ios::fixed);
	pov.precision(5);

	// Normals
	Vector* N = new Vector[nsn];
	Vector Ntmp;
	for (int i = 0; i < faces.size(); i++) {
		Ntmp = (Ut[faces[i].n2] - Ut[faces[i].n1]).cross(Ut[faces[i].n3] - Ut[faces[i].n1]);
		N[SNb[faces[i].n1]] += Ntmp;
		N[SNb[faces[i].n2]] += Ntmp;
		N[SNb[faces[i].n3]] += Ntmp;
	}
	for (int i = 0; i < nsn; i++) N[i].normalize();
	
	pov << "#include \"colors.inc\"\n";
	pov << "background { color rgb <1,1,1> }\n";
	pov << "camera { location <" << -1.5*zoom << ", " << 1.5*zoom << ", " << -1.5*zoom << "> look_at <0, 0, 0> sky <0, 0, -1> focal_point <-0.55, 0.55, -0.55> aperture 0.055 blur_samples 10 }\n";
	pov << "light_source { <-14, 3, -14> color rgb <1, 1, 1> }\n";
	
	pov << "intersection {\n";
	pov << "mesh2 { \n";
	pov << "vertex_vectors { " << nsn << ",\n";
	for (int i = 0; i < nsn; i++) {
		pov << "<" << Ut[SN[i]].x << "," << Ut[SN[i]].y << "," << Ut[SN[i]].z << ">,\n";
	}
	pov << "} normal_vectors { " << nsn << ",\n";
	for (int i = 0; i < nsn; i++) {
		pov << "<" << N[i].x << "," << N[i].y << "," << N[i].z << ">,\n";
	}
	pov << "} face_indices { " << faces.size() << ",\n";
	for (int i = 0; i < faces.size(); i++) {
		pov << "<" << SNb[faces[i].n1] << "," << SNb[faces[i].n2] << "," << SNb[faces[i].n3] << ">,\n"; 
	}
	pov << "} inside_vector<0,1,0> }\n";
	pov << "box { <-100, -100, -100>, <100, 100, 100> }\n";
	pov << "pigment { rgb<1,1,0.5> } normal { bumps 0.05 scale 0.005 } finish { phong 1 reflection 0.05 ambient 0 diffuse 0.9 } }\n";
		
	pov.close();
	delete [] N;
}
