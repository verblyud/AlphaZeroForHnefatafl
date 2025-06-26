#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]

mod connect4game;
mod con4_mcts;

use crate::connect4game::ai::*;
use crate::connect4game::game::chip;
use crate::connect4game::game::chip::{Chip};
use crate::connect4game::games::*;
use crate::connect4game::game::{Game, BoardState, PlayerType, Board, Player};
use crate::connect4game::io::{GameIO, TermIO};


use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;

use pyo3::prelude::*;
use pyo3::exceptions::PyKeyboardInterrupt;
use tch::CModule;


use std::ffi::CString;
use std::os::raw::c_char;
use winapi::um::libloaderapi::LoadLibraryA;

fn main() {


    let path = CString::new("F:/cancer/SciComp/en312/Lib/site-packages/torch/lib/torch_cuda.dll").unwrap();
    
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }


    let logger_path = "F:/cancer/SciComp/SciComp_Seminar_AlphaZero/agents/test/test_logs";
    let nn_path = "F:/cancer/SciComp/SciComp_Seminar_AlphaZero/agents/test/models/gen0.pt";

    // Create/open log file
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logger_path)
        .map_err(|e| format!("Failed to open log file: {}", e))
        .unwrap();



    let c_puct_base = 19652.0;
    let c_puct_init = 1.25;
    let root_noise = false; // Set to true if you want to use root noise
    let deterministic = true; // Set to true if you want to use deterministic search
    let warmup = false;

    let mcts_iterations = 800; // Number of MCTS iterations per move
    let no_games = 10; // Number of games to play


    let mut nnmodel = 
        if tch::Cuda::is_available() {
            CModule::load_on_device(nn_path, tch::Device::Cuda(0)).unwrap()
        } else {
            CModule::load_on_device(nn_path, tch::Device::Cpu).unwrap()
        };
    
    nnmodel.set_eval();
    let nnmodel = Arc::new(nnmodel);


    let board = Board::new(7,6);

    let players = vec![
    Player {
        player_type: PlayerType::AlphaZero,
        chip_options: vec![RED_CHIP],
        win_conditions: vec![four_in_a_row(RED_CHIP)],
    },
    Player {
        player_type: PlayerType::AlphaZero,
        chip_options: vec![YELLOW_CHIP],
        win_conditions: vec![four_in_a_row(YELLOW_CHIP)],
    },
    ];

    let mut game = Game::new(board, players);

    let io = TermIO::new();

    let mut move_count = 0;


    let mut root_node = con4_mcts::Node::new(1, //player
    1.0, //prior
    0, //action
    None,
    Some(0.0)); 
    


    writeln!(log_file, "=== Starting Connect4 Self-Play ===").unwrap();
    writeln!(log_file, "MCTS iterations: {}", mcts_iterations).unwrap();

    loop {
        io.draw_board(game.get_board());

        // Prompt user for input
        println!("Enter a move (0-6) or press Enter to enable AI mode:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read input");
        let input = input.trim();
        let ai = input.is_empty(); // If user just pressed Enter, enable AI mode

        


        
        if ai == true {
            writeln!(log_file, "AI mode enabled. Starting MCTS...").unwrap();

            let (action,_policy,_child_Q,_root_value,new_root_node) = con4_mcts::con4_search(
                    game.clone(),
                    &mut root_node,
                    &*nnmodel,
                    c_puct_base,
                    c_puct_init,
                    mcts_iterations as u32,
                    root_noise,
                    warmup,
                    deterministic,
                    &logger_path);
        
            root_node = (*new_root_node.borrow()).clone();
                        
            let player = match game.current_player().chip_options[0] {
                RED_CHIP => 1, // Red
                YELLOW_CHIP => -1, // Yellow
                _ => panic!("Invalid player chip option"),
            };

            let chip = game.current_player().chip_options[0];
            let chip_owner = match game.current_player().chip_options[0] {
                RED_CHIP => "Red", // Red
                YELLOW_CHIP => "Yellow", // Yellow
                _ => panic!("Invalid player chip option"),
            };

            game.play_no_check(action as isize, chip);

        } else {

        // If user provided input, parse it as an integer
        let input: isize = match input.parse() {
            Ok(num) if num >= 0 && num < game.get_board().width as isize => num,
            _ => {
                println!("Invalid input. Please enter a number between 0 and {}.", game.get_board().width - 1);
                continue;
            }
        };
        let player = game.current_player();
        let chip = player.chip_options[0];
        game.play_no_check(input, chip);

    // Create a fresh root node for the new game state after manual move
        let current_player = match game.current_player().chip_options[0] {
            RED_CHIP => 1,
            YELLOW_CHIP => 2,
            _ => panic!("Invalid player chip option"),
        };
        
        root_node = con4_mcts::Node::new(
            current_player, // Current player after the manual move
            1.0,           // Prior probability
            input as u32,  // The action that was just played
            None,          // No parent (this is the new root)
            Some(0.0)      // Initial value
        );
        }

    match game.compute_board_state() {
            BoardState::Ongoing => {
                move_count += 1;
                writeln!(log_file, "Move {}: Player {}", move_count, game.get_turn() + 1).unwrap();
            },
            BoardState::Win(player) => {
                io.draw_board(game.get_board());
                writeln!(log_file, "Player {} wins!", player + 1).unwrap();
                break;
            },
            BoardState::Draw => {
                io.draw_board(game.get_board());
                writeln!(log_file, "The game is a draw!").unwrap();
                break;
            },
            BoardState::Invalid => {
                    writeln!(log_file, "Invalid game state encountered. Continuing...").unwrap();
                    println!("Invalid game state encountered. Try again.");
                    continue;
            },
        }
    }
    println!("Game Over!");
    writeln!(log_file, "\n=== Self-Play Complete ===").unwrap();
    io.draw_board(game.get_board());

}

