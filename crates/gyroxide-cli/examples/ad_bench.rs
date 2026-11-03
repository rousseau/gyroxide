//! Cost of reverse-mode automatic differentiation through the explicit dynamics.
//!
//!     cargo run --release --features autodiff --example ad_bench -- <config.toml> <mesh> 1 10 50
//!
//! For each K: K forward steps on an autodiff wgpu device, a scalar loss on the final
//! positions, then `backward()`; prints forward and backward wall times and the
//! same K steps on the plain backend for reference. The gradient is taken with
//! respect to the initial positions (the cost does not depend on which input).
use burn::tensor::{Device, DeviceKind};
use gyroxide_mesh::netgen;
use gyroxide_model::Params;
use gyroxide_sim::Simulation;
use std::time::Instant;

fn forward(params: &Params, mesh: &gyroxide_mesh::TetMesh, k: usize) -> f64 {
    let mut sim = Simulation::new(mesh.clone(), params.clone(), Device::wgpu(DeviceKind::DefaultDevice)).unwrap();
    sim.step(); // warm-up (contact lists, kernel compilation)
    let t0 = Instant::now();
    for _ in 0..k {
        sim.step();
    }
    let _ = sim.positions(); // synchronise
    t0.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let params = Params::load(&args[1]).unwrap();
    let mesh = netgen::read(&args[2]).unwrap();
    let ks: Vec<usize> = args[3..].iter().map(|s| s.parse().unwrap()).collect();
    println!("{} nodes, {} tets", mesh.n_nodes(), mesh.n_tets());
    for k in ks {
        let plain = forward(&params, &mesh, k);
        let device = Device::wgpu(DeviceKind::DefaultDevice).autodiff();
        let mut sim = Simulation::new(mesh.clone(), params.clone(), device).unwrap();
        sim.step();
        let x0 = sim.x.x.clone().require_grad();
        sim.x.x = x0.clone();
        let t0 = Instant::now();
        for _ in 0..k {
            sim.step();
        }
        let loss = sim.x.x.clone().sum() + sim.x.y.clone().sum() + sim.x.z.clone().sum();
        let _ = loss.clone().into_scalar::<f32>();
        let fwd = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let grads = loss.backward();
        let g = x0.grad(&grads).expect("gradient");
        let gn: f32 = g.abs().sum().into_scalar();
        let bwd = t1.elapsed().as_secs_f64();
        println!(
            "K = {k:4}: plain {:.1} ms/step | autodiff forward {:.1} ms/step, backward {:.1} ms/step | |grad|_1 = {gn:.3e}",
            1e3 * plain / k as f64,
            1e3 * fwd / k as f64,
            1e3 * bwd / k as f64
        );
    }
}
