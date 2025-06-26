#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::hnefgame::game::state::GameState;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::game::{SmallBasicGame, Game};
use crate::hnefgame::game::GameOutcome::{Win, Draw};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use crate::hnefgame::preset::{boards, rules};
use crate::hnefgame::board::state::BoardState;
use crate::mcts::mcts;
use crate::support::{action_to_str, board_to_matrix, get_ai_play,write_to_file};
use crate::mcts_cmp::{MCTSAlg, mcts_do_alg};
use crate::mcts_rewritten;

use rand::prelude::*;
use rand::thread_rng;
use rand::distributions::WeightedIndex;
use tch::CModule;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{self, TryRecvError};
use std::str::FromStr;


fn generate_training_example<T: BoardState>(
    game_state_history: &Vec<GameState<T>>,
    policy_history: &Vec<Vec<f32>>,
    final_outcome: i32,
) -> Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)> {
    let mut training_examples = Vec::new();

    for (state, policy) in game_state_history.iter().zip(policy_history.iter()) {
        training_examples.push((
            board_to_matrix(state),
            policy.clone(),
            match state.side_to_play {
                Side::Attacker => 1,
                Side::Defender => -1,
            },
            final_outcome,
        ));
    }
    training_examples
}

pub fn self_play(
    nnmodel: Arc<CModule>,
    no_games: i32,
    mcts_iterations: usize,
    verbose: bool,
    mcts_alg: &str,
    num_workers: usize,
    c_puct: f32,
    alpha: f64,
    eps: f32,
) -> Result<Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)>, String>{

    let mcts_alg = MCTSAlg::from_str(mcts_alg).expect("Invalid MCTS algorithm: Choose from mcts_mcts, mcts_par_mcts_notpar, mcts_par_mcts_par, mcts_par_mcts_root_par");

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
        println!("Game number: {}", i);
    
    // Create new game
        let mut game: SmallBasicGame = Game::new(
            rules::KOCH,
            boards::BRANDUBH,
        ).expect("Could not create game.");

        let mut policy_history = Vec::new();

        loop {
            match rx.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => {
                    println!("Exiting...");
                    user_input_thread.join().expect("Failed to join user input thread");
                    return Err("User exited".to_string());
                }
                _ => {}
            }

            let move_time = Instant::now();
            let player = game.state.side_to_play;

            if verbose {
                println!("Player: {:?}", player);
                println!("Board:");
                println!("{}", game.state.board);
            }

            // let policy = mcts(&nnmodel, &game, 100);
            let policy = mcts_do_alg(
                &mcts_alg,
                Arc::clone(&nnmodel),
                &game,
                mcts_iterations,
                num_workers,
                c_puct,
                alpha,
                eps);
            policy_history.push(policy.clone());

            let mut rng = thread_rng();
            let dist = WeightedIndex::new(&policy).expect("Invalid distribution");
            let action = dist.sample(&mut rng) as u32;
            let str_action: &str = &action_to_str(&action);
            let play = get_ai_play(str_action);
            
            if verbose {
                println!("{}", str_action);
            }

            if game.state_history.len() == 100 {
                println!("Game over. Draw: Max moves reached.");
                let mut training_examples = generate_training_example(&game.state_history, &policy_history, 0);
                training_data.append(&mut training_examples);
                break;
            }

            match game.do_play(play) {
                Ok(status) => {
                    println!("Move took: {:?}", move_time.elapsed());
                    if let Over(outcome) = status {
                        match outcome {
                            Draw(reason) => {
                                println!("Game over. Draw {reason:?}.");
                                let mut training_examples = generate_training_example(&game.state_history, &policy_history, 0);
                                training_data.append(&mut training_examples);
                                break;
                            }
                            Win(reason, side) => {
                                let outcome_value = match side {
                                    Side::Attacker => 1,
                                    Side::Defender => -1,
                                };
                                println!("Game over. Winner is {side:?} ({reason:?}).");
                                let mut training_examples = generate_training_example(&game.state_history, &policy_history, outcome_value);
                                training_data.append(&mut training_examples);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Invalid move ({e:?}). Try again.");
                    //let max_index = policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(index, _)| index as u32).expect("Policy vector is empty");
                    //let play = get_ai_play(&action_to_str(&action));
                    continue
                }
            }
        }
    }
    Ok(training_data)
}