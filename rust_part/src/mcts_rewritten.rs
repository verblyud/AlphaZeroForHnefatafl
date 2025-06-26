#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]

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
use rand::distributions::WeightedIndex;
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
    player: i32,
    prior: f32,
    action: Action,
    parent: Option<Rc<RefCell<Node>>>,
    is_expanded: bool,
    visits: f32,
    value: f32,
    vl_applied: i32, // virtual loss applied
    children: HashMap<Action, Rc<RefCell<Node>>>, 
}

#[allow(dead_code)]
impl Node{
    fn new(player: i32, prior: f32, action: Action, parent: Option<Rc<RefCell<Node>>>) -> Self {
        Node {
            children: HashMap::new(),
            player,
            prior,
            action,
            parent,
            is_expanded: false,
            visits: 0.0, // visits start at 0
            value: 0.0, // value starts at 0
            vl_applied: 0 // virtual loss applied
        }
    }


    fn Q_value(&self) -> f32 {
        if self.visits == 0.0 {
            return 0.0; // If no visits, return 0
        }
        self.value / self.visits
    }

    fn has_parent(&self) -> bool {
        self.parent.is_some()
    }

    fn child_UCT_value(&self, c_puct_base:f32, c_puct_init: f32) -> Vec<f32> {
        if self.visits == 0.0 {
            panic!("Cannot compute UCT value when node has zero visits");
        }
        let pb_c = ((1.0 + self.visits + c_puct_base) / c_puct_base).ln() + c_puct_init;

        let parent_visits = self.parent.as_ref().map_or(1.0, |p| p.borrow().visits);
        self.children
            .values()
            .map(|child| {
                let child = child.borrow();
                let q = child.value;
                let p = child.prior;
                let n = child.visits;
                pb_c * p * ((parent_visits.sqrt()) / (1.0 + n))
            })
            .collect::<Vec<f32>>()
    }

    fn child_Q_value(&self) -> Vec<f32> {
        self.children
            .values()
            .map(|child| child.borrow().Q_value())
            .collect::<Vec<f32>>()
    }

    fn child_visits(&self) -> Vec<f32> {
        self.children
            .values()
            .map(|child| child.borrow().visits)
            .collect::<Vec<f32>>()
    }

}


fn choose_best_child(node: &mut Node, c_puct_base: f32, c_puct_init: f32, legal_moves: &Vec<bool>) -> Rc<RefCell<Node>> {

    if node.is_expanded == false {
        panic!("Expand Node first"); 
    }

    // vector of all actions, legal or not with their UCT values
    let neg_q_values: Vec<f32> = node.child_Q_value().iter().map(|x| -x).collect();
    let uct_values = node.child_UCT_value(c_puct_base, c_puct_init);
    let ucb_scores: Vec<f32> = neg_q_values.iter().zip(uct_values.iter()).map(|(a, b)| a + b).collect(); //since children Q values are from opponent's perspective, we need to negate them

    let masked_ucb_scores: Vec<f32> = ucb_scores
        .iter()
        .zip(legal_moves.iter())
        .map(|(score, &is_legal)| if is_legal { *score } else { f32::NEG_INFINITY })
        .collect();

    let action_id = masked_ucb_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index);

    if let Some(action_id) = action_id {
        if let Some(child) = node.children.get(&(action_id as Action)) {
            return Rc::clone(child);
        } else {
            panic!("Child not found in the node's children");
        }
    } else {
        panic!("No legal actions available");
    }
}


fn expand(node: &mut Node, prior_probs: Vec<f32>, child_to_play: i32) -> () {

    if node.is_expanded {
        panic!("Node is already expanded");
    }
    if prior_probs.len() == 0 {
        panic!("Prior probabilities are empty");
    }

    let num_children = prior_probs.len();
    let parent_rc = Rc::new(RefCell::new(node.clone()));
    for i in 0..num_children {
        let action = i as Action;
        let child_node = Node::new(child_to_play as i32, prior_probs[i], action, Some(parent_rc.clone()));
        node.children.insert(action, Rc::new(RefCell::new(child_node)));
    }
    node.is_expanded = true;
}


fn backpropagate(node: &mut Node, value: f32) -> () {
    let mut value = value;
    // Start with a raw pointer to the node (not recommended in general, but safe here since we control the tree)
    let mut current_node_ptr: Option<Rc<RefCell<Node>>> = node.parent.clone();
    while let Some(parent_rc) = current_node_ptr {
        {
            let mut parent_node = parent_rc.borrow_mut();
            parent_node.visits += 1.0;
            parent_node.value += value;
            // Prepare for next iteration
            current_node_ptr = parent_node.parent.clone();
        }
        value = -value;

    }
}

fn add_dirchlet_noise(node: &mut Node, legal_action: Vec<bool>, dirichlet_alpha: f32, eps:f32) -> () {
    if !node.is_expanded {
        panic!("Node is not expanded");
    }

    let mut rng = rand::thread_rng();
    let noise: Vec<f32> = legal_action
        .iter()
        .enumerate()
        .map(|(i, &is_legal)| {
            if is_legal {
                let noise_value: f32 = rng.gen_range(0.0..1.0);
                return noise_value;
            } else {
                return 0.0;
            }
        })
        .collect();

    let noise_sum: f32 = noise.iter().sum();
    let noise: Vec<f32> = noise.iter().map(|&x| x / noise_sum).collect();

    for (action, child) in &mut node.children {
        if legal_action[*action as usize] {
            let prior = child.borrow_mut().prior;
            let new_prior = (1.0 - eps) * prior + eps * noise[*action as usize];
            child.borrow_mut().prior = new_prior;
        }
    }
}

fn generate_search_policy(mut visit_counts: Vec<f32>, temperature: f32) -> Vec<f32> {

    if temperature > 0.0 {
        // Clamp exponent between 1.0 and 5.0
        let exp = 1.0 / temperature;
        let exp = exp.clamp(1.0, 5.0);
        for v in &mut visit_counts {
            *v = v.powf(exp);
        }
    }

    let sum: f32 = visit_counts.iter().sum();
    if sum > 0.0 {
        for v in &mut visit_counts {
            *v /= sum;
        }
    } else {
        // fallback to uniform if all are zero
        let n = visit_counts.len() as f32;
        for v in &mut visit_counts {
            *v = 1.0 / n;
        }
    }
    visit_counts
}


// Current one.
fn search<T: BoardState>(
    game_state: GameState<T>, 
    root_node: &mut Node, 
    nnmodel: &CModule, 
    game_logic: &GameLogic, 
    // Parameters
    c_puct_base: f32,
    c_puct_init: f32, 
    num_simulations: u32, 
    root_noise: bool, 
    warmup: bool, 
    determenistic: bool) -> (u32, Vec<f32>, f32, f32, Rc<RefCell<Node>>) {

    let current_player = match game_state.side_to_play {
        Side::Attacker => 1,
        Side::Defender => -1,
    };

    if root_noise {
        let legal_actions_i8 = generate_tile_plays(game_logic, &game_state);
        let legal_actions: Vec<bool> = legal_actions_i8.iter().map(|&x| x != 0).collect();
        add_dirchlet_noise(root_node, legal_actions, 0.03, 0.25);
    }


    // Wrap the root node in Rc<RefCell<Node>> for tree traversal
    let root_rc = Rc::new(RefCell::new(root_node.clone()));

    while root_rc.borrow().visits < num_simulations as f32 {
        let mut node_rc = Rc::clone(&root_rc);
        let mut game_state_copy = game_state.clone();
        let mut value: f32 = 0.0;
        let mut over = false;

        let mut loop_game_state = game_state_copy.clone();

        // Traverse the tree
        loop {
            let is_expanded = node_rc.borrow().is_expanded;
            if !is_expanded {
                break;
            }

            let legal_moves_i8 = generate_tile_plays(game_logic, &loop_game_state);
            let legal_moves: Vec<bool> = legal_moves_i8.iter().map(|&x| x != 0).collect();

            let next_node_rc = {
                            let mut node_ref_mut = node_rc.borrow_mut();
                            choose_best_child(&mut *node_ref_mut, c_puct_base, c_puct_init, &legal_moves)
                        };
            node_rc = next_node_rc;

            let play_string = action_to_str(&node_rc.borrow().action);
            let play: Play = get_ai_play(&play_string);
            let play_result = game_logic.do_play(play, loop_game_state).unwrap();
            loop_game_state = play_result.new_state;

            if loop_game_state.status != Ongoing {
                over = true;
                game_state_copy = loop_game_state.clone();
                break;
            }
        }
        if over {
            // If the game is over, we can backpropagate the value
            // 1 is assigned to the player who wins
            match game_state_copy.status {
                Ongoing => (),
                Over(outcome) => match outcome {
                    Win(_, side) => {
                        let side = match side {
                            Side::Attacker => 1.0,
                            Side::Defender => -1.0,
                        };
                        if node_rc.borrow().player as f64 == side {
                            value = 1.0;
                        } else {
                            value = -1.0;
                        }
                    }
                    Draw(_) => value = 0.0,
                },
            }
            backpropagate(&mut node_rc.borrow_mut(), value);
            continue;
        }

        let (prior_probs, value) = model_predict(&game_state_copy, nnmodel, game_logic);
        expand(&mut node_rc.borrow_mut(), prior_probs, -current_player);

        backpropagate(&mut node_rc.borrow_mut(), value);
    }
    let search_policy = if warmup {
        generate_search_policy(root_node.child_visits(), 1.0)
    } else {
        generate_search_policy(root_node.child_visits(), 0.1)
    };

    let action = match determenistic {
        true => { // choose the action with the highest visit count
            root_node.children
                .iter()
                .max_by(|(_, a), (_, b)| a.borrow().visits.partial_cmp(&b.borrow().visits).unwrap())
                .map(|(action, _)| *action)
                .expect("No children found in node")
        }
        false => { // sample an action based on the search policy
            let mut rng = thread_rng();
            let dist = WeightedIndex::new(&search_policy).expect("Invalid distribution");
            dist.sample(&mut rng) as u32
        }
        
    };

    let next_root = root_node.children.get(&action).expect("Action not found in children");
    next_root.borrow_mut().parent = None;

    let best_child_Q = -next_root.borrow().Q_value();

    return (action, search_policy, best_child_Q, root_node.value, next_root.clone());
}



fn model_predict<T: BoardState>
    (game_state: &GameState<T>, 
        nnmodel: &CModule, 
        _game_logic: &GameLogic) -> 
        (Vec<f32>, f32) 
    {
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

    if value < -1.0 || value > 1.0 {
        println!("Warning: value {} is out of range [-1, 1]", value);
    }

    let value = value.clamp(-1.0, 1.0);

    /* 
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
    */ 

    let mut pi: Vec<f32> = log_prob.iter()
        .map(|&p| p.exp()) // nnmodel outputs log_softmax, so we need to apply .exp() to recover the original probability
        .collect();

    let sum_probs: f32 = pi.iter().sum();

    if sum_probs > 0.0 {
        for p in &mut pi {
            *p /= sum_probs;
        }                           // renormalize
    } else {                                                // Contingency for when all the actions with non-zero probabilities are masked
        let n = pi.len() as f32;
        for p in &mut pi {
            *p = 1.0 / n; // fallback to uniform distribution
        }
    }
    return (pi, value)
}
