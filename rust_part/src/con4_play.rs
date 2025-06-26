#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]



use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;

use crate::connect4game::ai::*;
use crate::connect4game::game::chip;
use crate::connect4game::games::*;
use crate::connect4game::game::{Game, BoardState, PlayerType, Board, Player};
use crate::connect4game::game::chip::Chip;
use crate::connect4game::io::{GameIO, TermIO};
use crate::con4_mcts::con4_board_to_matrix_perspective;
use crate::con4_mcts;

use tch::CModule;
use std::time::Instant;
use std::thread;
use std::sync::mpsc::{self, TryRecvError};

pub fn self_play_connect4(
    nnmodel: Arc<CModule>,
    no_games: i32,
    mcts_iterations: usize,
    verbose: bool,
    num_workers: usize,
    c_puct: f32,
    alpha: f64,
    eps: f32,
    logger_path: &str
) -> Result<Vec<(Vec<Vec<i8>>, Vec<f32>, i32, i32)>, String>{

     // Create/open log file
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logger_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;

    writeln!(log_file, "=== Starting Connect4 Self-Play ===").unwrap();
    writeln!(log_file, "Games: {}, MCTS iterations: {}, Verbose: {}", no_games, mcts_iterations, verbose).unwrap();



    let c_puct_base = 19652.0;
    let c_puct_init = 1.25;
    let root_noise = false; // Set to true if you want to use root noise
    let deterministic = true; // Set to true if you want to use deterministic search
    let warmup = false;



    let (tx, rx) = mpsc::channel();
    let user_input_thread = thread::spawn(move || {
        let mut input = String::new();
        loop {
            std::io::stdin().read_line(&mut input).expect("Failed to read line");
            if input.trim() == "exit" {
                tx.send(()).expect("Failed to send exit signal");
                break;
            }
            input.clear();
        }
    });

    let mut training_data = Vec::new();

    for i in 0..no_games {
        writeln!(log_file, "\n--- Game {} ---", i).unwrap();
        println!("Game number: {}", i);
        // Create new game

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

        let mut policy_history = Vec::new();
        let mut game_history = Vec::new();
        let mut move_count = 0;


        let mut root_node = con4_mcts::Node::new(1, //player
            1.0, //prior
            0, //action
            None,
            Some(0.0)); 


        loop {
            match rx.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => {
                    writeln!(log_file, "User exit signal received").unwrap();
                    println!("Exiting...");
                    user_input_thread.join().expect("Failed to join user input thread");
                    return Err("User exited".to_string());
                }
                _ => {}
            }

            move_count += 1;
            writeln!(log_file, "Move {}: Player {}", move_count, 
                match game.current_player().chip_options[0] {
                    RED_CHIP => "RED",
                    YELLOW_CHIP => "YELLOW",
                    _ => "UNKNOWN",
                }).unwrap();

            
            if verbose {
                    let player = match game.current_player().chip_options[0] {
                        RED_CHIP => "Red", // Red
                        YELLOW_CHIP => "Yellow", // Yellow
                        _ => panic!("Invalid player chip option"),
                    };
                println!("Current player: {}", player);
            }

            io.draw_board(game.get_board());
            let move_time = Instant::now();
            writeln!(log_file, "Starting MCTS search with {} iterations...", mcts_iterations).unwrap();

            let search_start = Instant::now();

            // let policy = mcts(&nnmodel, &game, 100);
            let (action,policy,_child_Q,_root_value,new_root_node) = con4_mcts::con4_search(
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

            let search_duration = search_start.elapsed();
            writeln!(log_file, "MCTS search completed in {:?}, selected action: {}", search_duration, action).unwrap();


            // Check if search took too long
            if search_duration.as_secs() > 30 {
                writeln!(log_file, "WARNING: MCTS search took longer than 30 seconds!").unwrap();
            }
            


            root_node = (*new_root_node.borrow()).clone();
                
            policy_history.push(policy.clone());
            game_history.push(game.clone());

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
            writeln!(log_file, "Inserting chip {} in column {}", chip_owner, action).unwrap();
        
            game.play_no_check(action as isize, chip);


            if verbose {
                println!("Color: {}, Column:{}" , chip_owner, action);
            }

            match game.compute_board_state() {


                BoardState::Ongoing => {
                    let move_duration = move_time.elapsed();
                    writeln!(log_file, "Move completed in {:?}, game continues", move_duration).unwrap();
                    println!("Move took: {:?}", move_time.elapsed());
                    continue;
                },
                BoardState::Win(side) => {
                    let reward: i32;
                    if player == side as i32 {
                        reward = 1;
                    } else {
                        reward = -1;
                    }

                    writeln!(log_file, "Game over. Winner is player {} ({}) (reward: {})", player, chip_owner, reward).unwrap();
                    println!("Game over. Winner is player {player} ({chip_owner}).");
                    let mut training_examples = generate_training_example(
                        &game_history, 
                        &policy_history,
                        reward);
                    training_data.append(&mut training_examples);
                                        writeln!(log_file, "Generated {} training examples", training_examples.len()).unwrap();
                    break;
                }
                BoardState::Draw => {
                    writeln!(log_file, "Game over. Draw.").unwrap();
                    println!("Game over. Draw.");
                    let reward = 0;
                    let mut training_examples = generate_training_example(
                        &game_history, 
                        &policy_history,
                        reward);
                    training_data.append(&mut training_examples);
                    writeln!(log_file, "Generated {} training examples", training_examples.len()).unwrap();
                    
                    break;
                },
                BoardState::Invalid => {
                    writeln!(log_file, "Invalid game state encountered. Continuing...").unwrap();
                    println!("Invalid game state encountered. Try again.");
                    continue;
                },
            }
        }
    }
    writeln!(log_file, "\n=== Self-Play Complete ===").unwrap();
    writeln!(log_file, "Total training examples generated: {}", training_data.len()).unwrap();
    

    Ok(training_data)
}



pub fn generate_training_example(
    game_state_history: &Vec<Game>,
    policy_history: &Vec<Vec<f32>>,
    final_outcome: i32,
) -> Vec<(Vec<Vec<i8>>, Vec<f32>, i32, i32)> {
    let mut training_examples = Vec::new();

    for (state, policy) in game_state_history.iter().zip(policy_history.iter()) {
        training_examples.push((
            con4_board_to_matrix(state),
            policy.clone(),
            match state.current_player().chip_options[0] {
                RED_CHIP => 1, // Red
                YELLOW_CHIP => -1, // Yellow
                _ => panic!("Invalid player chip option"),
            } as i32,
            final_outcome,
        ));
    }
    training_examples
}



pub fn con4_board_to_matrix(game: &Game) -> Vec<Vec<i8>> {

    let layout = game.get_board_layout();

    let rows = 6;
    let cols = 7;

    // Initialize a 6x7 matrix filled with zeros
    let mut matrix = vec![vec![0; cols]; rows];

    // Ensure we don't exceed the layout bounds
    let max_index = std::cmp::min(layout.len(), rows * cols);

    // Iterate over the layout vector with bounds checking
    for index in 0..max_index {
        if let Some(chip) = &layout[index] {
            // Calculate the column (x) and row (y) based on the index
            let x = index % cols; // Column index
            let y = rows - 1 - (index / cols); // Row index (bottom to top)

            // Additional bounds checking to prevent panic
            if y < rows && x < cols {
                // Assign values to the matrix based on fg_color
                // 1 for red who go first, -1 for yellow who goes second
                if chip.fg_color == 1 {
                    matrix[y][x] = 1;
                } else if chip.fg_color == 3 {
                    matrix[y][x] = -1;
                }
            }
        }
    }

    matrix
}









pub fn play(game: &mut Game, io: impl GameIO) {
    let mut is_over = false;

    let mut root_node = con4_mcts::Node::new(1, //player
    1.0, //prior
    0, //action
    None,
    Some(1.0)); //parent

    while !is_over {
        io.draw_board(game.get_board());
        let (loc, ty) = match game.current_player().player_type {
            PlayerType::Local => io.get_move(game),
            PlayerType::Remote => io.get_move(game),
            PlayerType::AI(ai) => get_best_move(game, ai),
            PlayerType::AlphaZero => {

                let (action, policy, _child_Q, _root_value, new_root_node) = con4_mcts::con4_search(
                    game.clone(),
                    &mut root_node,
                    &*Arc::new(CModule::load("path_to_your_model.pt").unwrap()),
                    1.0,
                    1.0,
                    100,
                    false,
                    false,
                    false,
                    "path_to_your_logger.log"
                );

                root_node = (*new_root_node.borrow()).clone();
                (action as isize, game.current_player().chip_options[0])
            }
        };

        match game.play(loc, ty) {
            BoardState::Ongoing => {}
            BoardState::Invalid => {
                println!("\n\nInvalid move.");
            }
            x => {
                io.display_gameover(x);
                is_over = true;
            }
        }
    }
    io.draw_board(game.get_board());

    // for debugging
    game.print_moves();
    println!();
}





fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut game = if args.len() > 1 {
        match args[1].as_ref() {
            "toto" => toto(),
            "toto_ai" => toto_ai(),
            "3" => connect4_3player(),
            "ai" => connect4_ai(),
            "aibig" => connect4_large_ai(),
            "ai2" => connect4_ai_p2(),
            _ => connect4(),
        }
    } else {
        connect4()
    };
    play(&mut game, TermIO::new());
}
