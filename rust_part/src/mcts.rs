#![allow(unused_imports)]
#![allow(dead_code)]

use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::{Ongoing, Over};
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::play::Play;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::game::logic::GameLogic;
use super::support::{action_to_str, board_to_matrix, generate_tile_plays, get_ai_play, get_indices_of_ones, get_play};

use std::any::type_name;
use std::collections::HashMap;
use rand::prelude::*;
use tch::nn::{Module, ModuleT};
use tch::{CModule, Tensor, Kind, Device, IValue};
use std::cell::RefCell;
use std::rc::Rc;

type Action = u32;

const C_PUCT: f32 = 0.3;

// NOTE: We might not even need board field from Node.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Node {
    children: HashMap<Action, Rc<RefCell<Node>>>, 
    visits: f32,
    valid_actions: Vec<Action>, 
    action_probs: HashMap<Action, f32>,
    action_counts: HashMap<Action, f32>, 
    action_qs: HashMap<Action, f32>,
}

#[allow(dead_code)]
impl Node{
    fn new(valid_actions: Vec<Action>,
           visits: f32) -> Self {
        Node {
            children: HashMap::new(),
            visits,
            valid_actions,
            action_probs: HashMap::new(),
            action_counts: HashMap::new(),
            action_qs: HashMap::new(),
        }
    }

    fn add_child(&mut self, action: Action, child: Node) {
        self.children.insert(action, Rc::new(RefCell::new(child)));
    }

    fn uct_value(&self, action: &Action) -> f32 {
        // if self.action_counts(action) == 0.0 {
        //     return f64::INFINITY;
        // }
        // To do; add some dirichlet noise
        let q = self.action_qs.get(action).unwrap();
        let count = self.action_counts.get(action).unwrap();
        let p = self.action_probs.get(action).unwrap();
        return *q + C_PUCT * *p * (self.visits.ln() / (1.0 + *count)).sqrt();
    }

}

// Current one.
fn search<T: BoardState>(game_state: GameState<T>, node: &mut Node, nnmodel: &CModule, game_logic: &GameLogic) -> f32 {

    let current_player = game_state.side_to_play;

    match game_state.status {
        Ongoing => (),
        Over(outcome) => match outcome {
            Win(_, side) => {
                if current_player == side {
                    let reward = 1.0;
                    return -1.0 * reward;
                } else {
                    let reward = -1.0;
                    return -1.0 * reward;
                }
            }
            Draw(_) => return 0.0,
        },
    }


    // TODO: add a logic for when there IS no valid_actions
    // TEMPORARY SOLUTION

    if node.valid_actions.is_empty() {
        let reward = -1.0;
        return -1.0 * reward;
    }
    
    let current_valid_actions = node.valid_actions.clone();
    let action = current_valid_actions
            .iter()
            .max_by(|a, b| {
                node.uct_value(a).partial_cmp(&node.uct_value(b)).unwrap()
            })
            .unwrap();

    // // For testing purpose TEMP.
    // println!("{}", action);
    
    let play_string = action_to_str(action);
    let play: Play = get_ai_play(&play_string);

    let play_result = game_logic.do_play(play,game_state).unwrap(); //check this again
    let next_state = play_result.new_state;

    if !node.children.contains_key(action) {         
        let reward = expand(node, &action, &next_state, &nnmodel, &game_logic);
        return -1.0 * reward
    }

    let next_node = node.children.get_mut(action).unwrap();
    let reward = search(next_state, &mut next_node.borrow_mut(), &nnmodel, game_logic);

    if let Some(q) = node.action_qs.get_mut(action) {
        if let Some(count) = node.action_counts.get(action) {
            *q = (*count * *q + reward) / (*count + 1.0); //dangerous spot
        }
    }

    if let Some(result) = node.action_counts.get_mut(action) {
        *result += 1.0;
    }
    node.visits += 1.0;
    return -1.0 * reward
    
}

fn expand<T: BoardState>(parent: &mut Node, action: &Action, game_state: &GameState<T>, nnmodel: &CModule, game_logic: &GameLogic) -> f32 {
    let (valid_actions, pi, value) = model_predict(game_state, nnmodel, game_logic);
    let num_valid_actions = valid_actions.len();
    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut action_qs = HashMap::with_capacity(num_valid_actions);
    let mut action_probs = HashMap::with_capacity(num_valid_actions);

    if !valid_actions.is_empty() {
        for action in valid_actions.iter() {
            action_counts.insert(*action, 0.0);
            action_qs.insert(*action, 0.0);
            action_probs.insert(*action, pi[*action as usize]);
        }
    }

    let new_node: Node = Node{
        children: HashMap::new(),
        visits: 1.0,
        valid_actions,
        action_probs,
        action_counts,
        action_qs,
    };

    parent.add_child(*action, new_node);
    return value
}

fn model_predict<T: BoardState>(game_state: &GameState<T>, nnmodel: &CModule, game_logic: &GameLogic) -> (Vec<u32>, Vec<f32>, f32) {

    // Preparing input Tensors
    // let matrix_representation = board_to_matrix(game_state);
    let matrix_representation: Vec<Vec<f32>> = board_to_matrix(game_state)
    .iter()
    .map(|row| row.iter().map(|&x| x as f32).collect())
    .collect();

    let player = match game_state.side_to_play {
        Side::Attacker => 1,
        Side::Defender => -1,
    };

    let board = Tensor::from_slice2(&matrix_representation); //Assuming game.state is a 2-dimensional vector
    let cond: [bool; 1] = if player == 1 {[true]} else {[false]};
    let cond: Tensor = Tensor::from_slice(&cond);

    // Run the inference using the nnmodel
    // let input = IValue::Tuple(vec![IValue::Tensor(board), IValue::Tensor(cond)]);
    // let output = nnmodel.forward_is(&[input]);
    
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let board = board.to_device(device);
    let cond = cond.to_device(device);
    let output = nnmodel.forward_is(&[IValue::Tensor(board), IValue::Tensor(cond)]);
    let (log_prob, value) = match output {
        Ok(IValue::Tuple(output)) => {
            if output.len() != 2 {
                panic!("Expected tuple of 2 tensors, but got {}", output.len());
            }

            let out1 = match &output[0] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => panic!("Expected a tensor as the first output"),
            };

            let out2 = match &output[1] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => panic!("Expected a tensor as the second output"),
            };

            (out1, out2)
        },
        _ => panic!("unexpected output from the model"),
    };

    // Converting outputs into vectors
    let log_prob = log_prob.flatten(0, i64::try_from(log_prob.size().len()).unwrap() - 1);
    let log_prob = Vec::<f32>::try_from(log_prob).expect("Something went wrong when converting tensor into vector");
    let value = value.flatten(0, i64::try_from(value.size().len()).unwrap() - 1);
    let value = f32::try_from(value).expect("Could not convert value tensor to f32");

    // NOTE: The NN model is trained to always predict the value from the Attacker's perspective. 
    // Therefore, in MCTS, we need to adjust the value to the current player's perspective.

    let value = match player {
        1 => value,
        -1 => -value,
        _ => panic!("Invalid player value"),
    };

    let valid_actions_for_masking: Vec<i8> = generate_tile_plays(game_logic, game_state); 
    // This should output a correct valid moves depending on the variable game (-> whose turn it is)  


    let valid_actions = get_indices_of_ones(&valid_actions_for_masking);
    let valid_actions: Vec<u32> = valid_actions.iter().map(|&x| x.try_into().expect("could not convert action into u32")).collect();

    if valid_actions.is_empty() {
        return (valid_actions, Vec::new(), 0.0)
    }

    let mut pi: Vec<f32> = log_prob.iter()
        .zip(valid_actions_for_masking.iter())
        .map(|(p, v)| p.exp() * (*v as f32))  // nnmodel outputs log_softmax, so we need to apply .exp() to recover the original probability
        .collect();

    let sum_probs: f32 = pi.iter().sum();

    if sum_probs > 0.0 {
        for p in &mut pi {
            *p /= sum_probs;
        }                           // renormalize
    } else {                                                // Contingency for when all the actions with non-zero probabilities are masked
        let num_valid_actions = valid_actions.len();
        pi = valid_actions_for_masking.iter().map(|&v| v as f32 / num_valid_actions as f32).collect();
    }

    return (valid_actions, pi, value)
}

fn get_improved_policy(root: Node) -> Vec<f32> {
    let pi_size = 7_usize.pow(4); //                NOTE: Assuming the board is 7x7 if not, change this
    let mut pi = vec![0.0; pi_size];
    let count_sum: f32 = root.children.values().map(|child| child.borrow().visits).sum();
    for (action, child) in &root.children {
        // let index = action_to_index(action); 
        let index: usize = *action as usize;
        pi[index] = child.borrow().visits / count_sum;
    }
    pi
}

// This does a single mcts starting from whomever the current turn is assigned to  
pub fn mcts<T: BoardState>(nnmodel: &CModule, game: &Game<T>, iterations: usize) -> Vec<f32> {
    let game_logic: GameLogic = game.logic;

    let (valid_actions, pi, _value) = model_predict(&game.state, &nnmodel, &game_logic);
    let num_valid_actions = valid_actions.len();

    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut action_qs = HashMap::with_capacity(num_valid_actions);
    let mut action_probs = HashMap::with_capacity(num_valid_actions);

    for action in valid_actions.iter() {
        action_counts.insert(*action, 0.0);
        action_qs.insert(*action, 0.0);
        action_probs.insert(*action, pi[*action as usize]);
    }

    let mut root = Node {
        children: HashMap::with_capacity(num_valid_actions),
        visits: 1.0,
        valid_actions,
        action_probs,
        action_qs,
        action_counts,
    };

    for _ in 0..iterations {
        let game_state_copy = game.state.clone();
        let _ = search(game_state_copy, &mut root, &nnmodel, &game_logic);
    }


    get_improved_policy(root)
}
