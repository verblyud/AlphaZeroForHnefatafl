pub mod self_play;
pub mod support;
pub mod mcts;
pub mod mcts_par;
pub mod mcts_cmp;
pub mod mcts_rewritten;

pub mod hnefgame {
    pub mod board;
    pub mod game;
    pub mod bitfield;
    pub mod error;
    pub mod pieces;
    pub mod play;
    pub mod preset;
    pub mod rules;
    pub mod tiles;
    pub mod utils;
}
pub mod con4_play;
pub mod con4_mcts;
pub mod connect4game;

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::PyKeyboardInterrupt;
use tch::CModule;


use std::ffi::CString;
use std::os::raw::c_char;
use winapi::um::libloaderapi::LoadLibraryA;


#[pyfunction]
fn hnef_self_play_function<'py> (nnmodel_path: &str, no_games: i32, mcts_iterations: usize, verbose: bool, mcts_alg: &str, num_workers: usize, c_puct: f32, alpha: f64, eps: f32) 
-> PyResult<Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)>> {

    let path = CString::new("F:/cancer/SciComp/en312/Lib/site-packages/torch/lib/torch_cuda.dll").unwrap();
    
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }


    let mut nnmodel = 
    if tch::Cuda::is_available() {
        CModule::load_on_device(nnmodel_path, tch::Device::Cuda(0)).unwrap()
    } else {
        CModule::load_on_device(nnmodel_path, tch::Device::Cpu).unwrap()
    };
    
    nnmodel.set_eval();
    let nnmodel = Arc::new(nnmodel);
    let result = self_play::self_play(nnmodel, no_games, mcts_iterations, verbose, mcts_alg, num_workers, c_puct, alpha, eps);
    match result {
        Ok(training_data) => Ok(training_data),
        Err(e) => Err(PyErr::new::<PyKeyboardInterrupt, _>(e)),
    }

}

#[pyfunction]
fn con4_self_play_function<'py> (nnmodel_path: &str, no_games: i32, mcts_iterations: usize, verbose: bool, num_workers: usize, c_puct: f32, alpha: f64, eps: f32, logger_path: &str) 
-> PyResult<Vec<(Vec<Vec<i8>>, Vec<f32>, i32, i32)>> {

    let path = CString::new("F:/cancer/SciComp/en312/Lib/site-packages/torch/lib/torch_cuda.dll").unwrap();
    
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }


    let mut nnmodel = 
    if tch::Cuda::is_available() {
        CModule::load_on_device(nnmodel_path, tch::Device::Cuda(0)).unwrap()
    } else {
        CModule::load_on_device(nnmodel_path, tch::Device::Cpu).unwrap()
    };
    
    nnmodel.set_eval();
    let nnmodel = Arc::new(nnmodel);
    let result = con4_play::self_play_connect4(nnmodel, no_games, mcts_iterations, verbose, num_workers, c_puct, alpha, eps, logger_path);
    match result {
        Ok(training_data) => Ok(training_data),
        Err(e) => Err(PyErr::new::<PyKeyboardInterrupt, _>(e)),
    }

}



#[pymodule]
#[pyo3(name="_azhnefatafl")]
fn _azhnefatafl(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(hnef_self_play_function, m)?)?;
    m.add_function(wrap_pyfunction!(con4_self_play_function, m)?)?;
    Ok(())
}