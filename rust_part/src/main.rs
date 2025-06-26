#![allow(unused_imports)]
#![allow(unused_variables)]


pub mod self_play;
pub mod support;
pub mod mcts;
pub mod hnefgame;
pub mod mcts_par;
pub mod mcts_cmp;

pub mod con4_play;
pub mod con4_mcts;
pub mod connect4game;
pub mod mcts_rewritten;


use hnefgame::game::{Game, SmallBasicGame};
use hnefgame::game::GameOutcome::{Draw, Win};
use hnefgame::game::GameStatus::Over;
use std::any::type_name;
use support::get_all_possible_moves;
use support::get_play;
// use mcts::mcts;
use self_play::self_play;




fn main() {
    println!("hnefatafl-rs demo");
    let mut game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");


    loop {


        println!("Board:");
        println!("{}", game.state.board);

        println!("{:?} to play.", game.state.side_to_play);

        //println!("Possible moves: {:?}", get_all_possible_moves(&game));

        let play = match get_play() {
            Some(play) => play,
            None => {
                println!("Exiting the game.");
                return;
            }
        };

        

        match game.do_play(play) {
            Ok(status) => {
                if let Over(outcome) = status {
                    match outcome {
                        Draw(reason) => println!("Game over. Draw {reason:?}."),
                        Win(reason, side) => println!("Game over. Winner is {side:?} ({reason:?})."),
                    }
                    println!("Final board:");
                    println!("{}", game.state.board);
                    return
                }
            },
            Err(e) => println!("Invalid move ({e:?}). Try again.")
        }
    }
}