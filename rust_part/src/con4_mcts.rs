#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]



// This does a single mcts starting from whomever the current turn is assigned to  
// Include all the necessary imports


use crate::connect4game::game::*;
use crate::connect4game::game::chip::Chip;
use crate::connect4game::games::{RED_CHIP, YELLOW_CHIP};

use std::collections::VecDeque;
use std::collections::HashMap;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use tch::nn::{Module, ModuleT};
use tch::{CModule, Tensor, Kind, Device, IValue};
use std::cell::RefCell;
use std::rc::Rc;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, Once};

// Global variables
static INIT: Once = Once::new();
static mut MCTS_LOG: Option<Mutex<std::fs::File>> = None;

// Helper function to initialize and get the global log
fn get_mcts_log() -> &'static Mutex<std::fs::File> {
    unsafe {
        MCTS_LOG.as_ref().expect("MCTS log not initialized. Call init_mcts_log() first.")
    }
}

// Function to initialize the log file (call this in con4_search)
fn init_mcts_log(logger_path: &str) {
    unsafe {
        if MCTS_LOG.is_some() {
            return; // Already initialized
        }
    }
    
    let mcts_log_path = if let Some(parent) = std::path::Path::new(logger_path).parent() {
        parent.join("MCTS_logger.txt")
    } else {
        std::path::PathBuf::from("MCTS_logger.txt")
    };
    
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&mcts_log_path)
        .expect("Failed to create MCTS log file");
    
    unsafe {
        MCTS_LOG = Some(Mutex::new(log_file));
    }
}

// Updated macro for safe logging
macro_rules! mcts_log {
    ($($arg:tt)*) => {
        unsafe {
            if let Some(ref log_mutex) = MCTS_LOG {
                if let Ok(mut log) = log_mutex.lock() {
                    writeln!(log, $($arg)*).unwrap();
                }
            }
        }
    };
}

#[allow(dead_code)]
type Action = u32;

// NOTE: We might not even need board field from Node.
#[allow(dead_code)]
#[derive(Debug, Clone,Default)]
pub struct Node {
    pub player: i32,
    pub prior: f32,
    pub action: Action,
    pub parent: Option<Rc<RefCell<Node>>>,
    pub is_expanded: bool,
    pub visits: f32,
    pub value: f32,
    pub v_loss: f32, // virtual loss
    pub vl_applied: i32, // count of virtual losses applied
    pub children: HashMap<Action, Rc<RefCell<Node>>>, 
}

#[allow(dead_code)]
impl Node{
    pub fn new(player: i32, prior: f32, action: Action, parent: Option<Rc<RefCell<Node>>>, visits: Option<f32>) -> Self {
        Node {
            children: HashMap::new(),
            player,
            prior,
            action,
            parent,
            is_expanded: false,
            visits: visits.unwrap_or(0.0), // visits start at 0
            value: 0.0, // value starts at 0
            v_loss: 0.0, // virtual loss starts at 0
            vl_applied: 0
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
        //mcts_log!("Computing UCT values with pb_c: {}, c_puct_base: {} and c_puct_init: {}", pb_c, c_puct_base, c_puct_init);

        let parent_visits = self.visits;

        let mut child_uct_vec: Vec<f32> = Vec::with_capacity(self.children.len());
        // Iterate over all actions (0 to 6 for Connect4)


        for action in 0..self.children.len() as u32 {
            if let Some(child) = self.children.get(&(action as Action)) {
                let child_borrowed = child.borrow();
                let p = child_borrowed.prior;
                let n = child_borrowed.visits;
                let uct_value = pb_c * p * ((parent_visits.sqrt()) / (1.0 + n));
                child_uct_vec.push(uct_value);
            } else {
                //mcts_log!("No child found for action {}", action);
            }

        }
        //mcts_log!("Computed UCT values: {:?}", child_uct_vec);
        child_uct_vec
    }

    fn child_Q_value(&self) -> Vec<f32> {

        let mut child_q_vec: Vec<f32> = Vec::with_capacity(self.children.len());

        // Iterate over all actions (0 to 6 for Connect4)
        for action in 0..self.children.len() as u32 {
            if let Some(child) = self.children.get(&(action as Action)) {
                let child_borrowed = child.borrow();
                let q_value = child_borrowed.Q_value();
                child_q_vec.push(q_value);
            } else {
                //mcts_log!("No child found for action {}", action);
            }
        }
        //mcts_log!("Computed Q values: {:?}", child_q_vec);
        child_q_vec

    }

    fn child_visits(&self) -> Vec<f32> {

        let mut child_visits_vec: Vec<f32> = Vec::with_capacity(self.children.len());
        // Iterate over all actions (0 to 6 for Connect4)

        for action in 0..self.children.len() as u32 {
            if let Some(child) = self.children.get(&(action as Action)) {
                let child_borrowed = child.borrow();
                let visits = child_borrowed.visits;
                child_visits_vec.push(visits);
            } else {
                //mcts_log!("No child found for action {}", action);
            }
        }
        //mcts_log!("Computed visits: {:?}", child_visits_vec);
        child_visits_vec
    }

}


fn choose_best_child(node: &mut Node, c_puct_base: f32, c_puct_init: f32, legal_moves: &Vec<bool>) -> Rc<RefCell<Node>> {
    //mcts_log!("Choosing best child from {} children", node.children.len());
    
    if node.is_expanded == false {
        panic!("Expand Node first"); 
    }

    // vector of all actions, legal or not with their UCT values
    let neg_q_values: Vec<f32> = node.child_Q_value().iter().map(|x| -x).collect();
    //mcts_log!("Negative Q values: {:?}", neg_q_values);
    let uct_values = node.child_UCT_value(c_puct_base, c_puct_init);
    //mcts_log!("UCT values: {:?}", uct_values);
    let ucb_scores: Vec<f32> = neg_q_values.iter().zip(uct_values.iter()).map(|(a, b)| a + b).collect(); //since children Q values are from opponent's perspective, we need to negate them
    //mcts_log!("UCB scores: {:?}", ucb_scores);

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
    //mcts_log!("Expanding node with {} children, child_to_play: {}", prior_probs.len(), child_to_play);
    
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
        let child_node = Node::new(child_to_play as i32, prior_probs[i], action, Some(parent_rc.clone()),Some(0.0));
        node.children.insert(action, Rc::new(RefCell::new(child_node)));
    }
    node.is_expanded = true;
    
    //mcts_log!("Node expansion complete. {} children added.", num_children);
}


fn backpropagate(root_rc: &Rc<RefCell<Node>>, mut value: f32, tree_traversal_path: Vec<u32>) -> () {

    if tree_traversal_path.is_empty() {
        //mcts_log!("No actions in traversal path, updating root.");
        root_rc.borrow_mut().value += value; // Update root value directly
        root_rc.borrow_mut().visits += 1.0; // Increment root visits
        return;
    }

    if tree_traversal_path.len() % 2 == 0 {
    } else {
        value = -value; // If the path length is odd, we negate the value
    }
    
    //mcts_log!("Root value updated from: {} to {}", root_rc.borrow().value, root_rc.borrow().value + value);
    root_rc.borrow_mut().value += value; // Update root value first

    root_rc.borrow_mut().visits += 1.0; // Increment root visits
    //mcts_log!("Root visits after initial update: {}", root_rc.borrow().visits);

    let mut update_rc = Rc::clone(root_rc);
    // Iterate through the actions in the traversal path

    //mcts_log!("Backpropagating through the tree traversal path {:?}", tree_traversal_path);
    for action in tree_traversal_path.iter() {
        
        // Get the child node
        let child_rc = {
            let node_borrowed = update_rc.borrow();
            node_borrowed.children.get(action)
                .expect("Action not found in children")
                .clone()
        };
        
        // Update the child's value and visits
        child_rc.borrow_mut().value += -value; // Negate value because backpropagation alternates perspective
        child_rc.borrow_mut().visits += 1.0;
        child_rc.borrow_mut().is_expanded = true; // Ensure the child is marked as expanded
    
        //mcts_log!("Backpropagating action {}: Updated value to {}, visits to {}", 
                 //action, 
                 //child_rc.borrow().value, 
                 //child_rc.borrow().visits);
        
        value = -value; // Flip value for next level
        
        // Move to the child node for next iteration
        update_rc = child_rc;
        
        //mcts_log!("Next node: {}", update_rc.borrow().action);
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

    //mcts_log!("Visit counts before temperature adjustment: {:?}", visit_counts);
    if temperature > 0.0 {
        // Clamp exponent between 1.0 and 5.0
        let exp = 1.0 / temperature;
        let exp = exp.clamp(1.0, 5.0);
        for v in &mut visit_counts {
            *v = v.powf(exp);
        }
    }
    //mcts_log!("Visit counts after temperature adjustment: {:?}", visit_counts);

    let sum: f32 = visit_counts.iter().sum();
    //mcts_log!("Sum of visit counts: {}", sum);
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
    //mcts_log!("Visit counts after normalization: {:?}", visit_counts);
    visit_counts
}





fn model_predict(game: &Game, game_history:VecDeque<Game> ,nnmodel: &CModule) -> (Vec<f32>, f32) {

    // Preparing input Tensors
    let current_player = match game.current_player().chip_options[0] {
        RED_CHIP => 1, // Red
        YELLOW_CHIP => -1, // Yellow
        _ => panic!("Invalid player chip option"),
    };

    // Create a vector to hold all board states (current + history)
    let mut all_game_states = Vec::new();
    
    // Add current game state first (X_t)
    all_game_states.push(game.clone());

    // Add historical states in reverse order (X_t-1, X_t-2, ..., X_t-7)
    for historical_game in game_history.iter().rev() {
        all_game_states.push(historical_game.clone());
        if all_game_states.len() >= 8 {
            break; // Only take the last 8 states
        }
    }

    // Create board representations for each state
    let mut all_boards = Vec::new();
    
    for game_state in &all_game_states {
        // Get board from current player's perspective
        let matrix_x = con4_board_to_matrix_perspective(game_state,current_player);
        // Get board from opponent's perspective  
        let matrix_y = con4_board_to_matrix_perspective(game_state,current_player);
        
        all_boards.push(matrix_x);
        all_boards.push(matrix_y);
    }

    // Pad with zeros if we have fewer than 8 states
    while all_boards.len() < 16 { // 8 states * 2 perspectives = 16 boards
        let empty_board = vec![vec![0.0f32; 7]; 6]; // 6x7 board of zeros
        all_boards.push(empty_board);
    }


        // Convert to tensor format: [batch_size, channels, height, width]
    // where channels = 16 (8 timesteps * 2 perspectives)
    let tensor_data: Vec<f32> = all_boards
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    
    // Reshape to [1, 16, 6, 7] - batch_size=1, channels=16, height=6, width=7
    let board_tensor = Tensor::from_slice(&tensor_data)
        .view([1, 16, 6, 7]);


    // Create condition tensor
    let cond_value = if current_player == 1 { 0.0f32 } else { 1.0f32 };
    let cond_data: Vec<f32> = vec![cond_value; 1 * 1 * 6 * 7]; // 42 elements
    let cond_tensor = Tensor::from_slice(&cond_data).view([1, 1, 6, 7]);
    
    // Move to device
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let board_tensor = board_tensor.to_device(device);
    let cond_tensor = cond_tensor.to_device(device);
    
    // Run inference
    let output = nnmodel.forward_is(&[IValue::Tensor(board_tensor), IValue::Tensor(cond_tensor)]);

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


    if value < -1.0 || value > 1.0 {
        println!("Warning: value {} is out of range [-1, 1]", value);
    }

    let value = value.clamp(-1.0, 1.0);


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


// Converts a board position into matrix to be fed to the neural network
// Each sell is either 1 if the chip is from the current player, 0 if empty or occupied by the opponent's chip
pub fn con4_board_to_matrix_perspective(game: &Game, player:i32) -> Vec<Vec<f32>> {

    let layout = game.get_board_layout();

    let rows = 6;
    let cols = 7;

    // Initialize a 6x7 matrix filled with zeros
    let mut matrix = vec![vec![0.0; cols]; rows];

    // Ensure we don't exceed the layout bounds
    let max_index = std::cmp::min(layout.len(), rows * cols);

    let target_color = if player == 1 { 
        RED_CHIP.fg_color 
    } else { 
        YELLOW_CHIP.fg_color 
    };

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
                if chip.fg_color == target_color {
                    matrix[y][x] = 1.0;
                } else {
                    matrix[y][x] = 0.0;
                }
            }
        }
    }
    matrix
}






fn negate_expanded_values(node_rc: &Rc<RefCell<Node>>) {
    let mut node = node_rc.borrow_mut();
    
    if node.is_expanded {
        node.value = -node.value;
        //mcts_log!("Negated value for expanded node (action {}): {}", node.action, node.value);
    }
    
    // Get children to traverse (need to collect to avoid borrow conflicts)
    let children: Vec<Rc<RefCell<Node>>> = node.children.values().cloned().collect();
    
    // Release the borrow before recursing
    drop(node);
    
    // Recursively negate children
    for child_rc in children {
        negate_expanded_values(&child_rc);
    }
}




// Add this function to your con4_mcts.rs file
fn visualize_tree(node: &Node, depth: usize, max_depth: usize, prefix: String, is_last: bool) -> String {
    if depth > max_depth {
        return String::new();
    }
    
    let mut result = String::new();
    
    // Current node info
    let connector = if depth == 0 {
        "Root"
    } else if is_last {
        "└──"
    } else {
        "├──"
    };
    
    let node_info = format!(
        "{}{} Action:{} Visits:{:.1} Value:{:.3} Q:{:.3} Prior:{:.3} Expanded:{}\n",
        prefix,
        connector,
        node.action,
        node.visits,
        node.value,
        node.Q_value(),
        node.prior,
        node.is_expanded
    );
    result.push_str(&node_info);
    
    // Children
    if node.is_expanded && depth < max_depth {
        let mut children: Vec<_> = node.children.iter().collect();
        children.sort_by_key(|(action, _)| **action);
        for (i, (action, child_rc)) in children.iter().enumerate() {
            let is_last_child = i == children.len() - 1;
            let new_prefix = if depth == 0 {
                String::new()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };
            
            let child = child_rc.borrow();
            result.push_str(&visualize_tree(&*child, depth + 1, max_depth, new_prefix, is_last_child));
        }
    }
    
    result
}

// Function to print tree to log
fn log_tree_visualization(root_node: &Node, iteration: usize, max_depth: usize) {
    //mcts_log!("\n=== TREE VISUALIZATION - Iteration {} ===", iteration);
    let tree_str = visualize_tree(root_node, 0, max_depth, String::new(), false);
    //mcts_log!("{}", tree_str);
    //mcts_log!("=== END TREE VISUALIZATION ===\n");
}


// Current one.
pub fn con4_search (
    game_state: Game, 
    root_node: &mut Node, 
    nnmodel: &CModule, 
    // Parameters
    c_puct_base: f32,
    c_puct_init: f32, 
    num_simulations: u32, 
    root_noise: bool, 
    warmup: bool, 
    determenistic: bool,
    logger_path: &str) -> (u32, Vec<f32>, f32, f32, Rc<RefCell<Node>>) {

    // Initialize the global log
    init_mcts_log(logger_path);
    
    //mcts_log!("\n=== Starting MCTS Search ===");
    //mcts_log!("Target simulations: {}", num_simulations);

    if root_noise {
        //mcts_log!("Adding Dirichlet noise to root...");
        let legal_actions_i8 = game_state.board.get_valid_moves();
        let legal_actions: Vec<bool> = (0..7)
            .map(|i| legal_actions_i8.contains(&(i as isize)))
            .collect();
        //mcts_log!("Legal actions: {:?}", legal_actions);
        add_dirchlet_noise(root_node, legal_actions, 0.03, 0.25);
        //mcts_log!("Dirichlet noise added.");
    }

    // Create the root_rc ONCE and use it throughout
    // root rc is a reference counted pointer to the root node passed by the play function
    let root_rc = Rc::new(RefCell::new(std::mem::take(root_node)));

    let mut simulation_count = 0;

    negate_expanded_values(&root_rc);
    
    // Now use root_rc for the condition and all operations
    while root_rc.borrow().visits < num_simulations as f32 {


        // here we are inside individual simulation
        // they start from the root node which is passed by the play function
        // these represent playout states from the root node board state
        // the root node value/visit counts are directly modified and then will be used to select the next best move

        let mut tree_traversal_path = Vec::new();

        simulation_count += 1;
        //mcts_log!("\n--- Simulation {} ---", simulation_count);
        //mcts_log!("Root visits: {}/{}", root_rc.borrow().visits, num_simulations);

        // Safety check for infinite loop
        if simulation_count as u32 > num_simulations * 2 {
            //mcts_log!("ERROR: Too many simulations! Breaking to prevent infinite loop.");
            break;
        }

        // Start from the root for this simulation - NO CLONING
        let mut node_rc = root_rc.clone(); // This creates a reference to the node that will be used in the simulation

        let mut value: f32 = 0.0;
        let mut over = false;

        // we dont want to modify the original game state since its the one where the game is actually played
        // so we create a copy of the game state to be used in the simulation
        let mut loop_game_state = game_state.clone();

        // Initialize deque to track game state history (last 8 moves)
        let mut loop_game_history: VecDeque<Game> = VecDeque::with_capacity(8);


        //mcts_log!("Starting tree traversal...");
        
        // Traverse the tree
        let mut traversal_depth = 0;

        

        loop {



            // tree traversal is done until we reach a node that is not expanded by following the best child
            // we then expand that leaf node and use model_predict to get prior probabilities and value for that state
            //traversal is also done directly on the original tree 

            // node_rc is a reference counted pointer to the current node in the tree

            loop_game_history.push_back(loop_game_state.clone());
            traversal_depth += 1;
            //mcts_log!("Traversal depth: {}", traversal_depth);
            
            // Safety check for infinite traversal
            if traversal_depth > 50 {
                //mcts_log!("ERROR: Traversal too deep! Breaking to prevent infinite loop.");
                break;
            }

            let is_expanded = node_rc.borrow().is_expanded;
            //mcts_log!("Current node {} expanded: {}",node_rc.borrow().action, is_expanded);
            
            // expected termination condition
            if !is_expanded {
                //mcts_log!("Node not expanded, breaking traversal");
                break;
            }
            
            ///////////////////////////////////
            // if expanded start next move selection
            let valid_indices = loop_game_state.board.get_valid_moves();

            let legal_actions: Vec<bool> = (0..7)
                .map(|i| valid_indices.contains(&(i as isize)))
                .collect();
            //mcts_log!("Legal actions at depth {}: {:?}", traversal_depth, legal_actions);

            // Check if any legal moves exist
            if !legal_actions.iter().any(|&x| x) {
                //mcts_log!("No legal moves available!");
                over = true;
                break;
            }

            // select the best child node based on UCT
            let next_node_rc = {
                let mut node_ref_mut = node_rc.borrow_mut();
                //mcts_log!("Choosing best child...");
                choose_best_child(&mut *node_ref_mut, c_puct_base, c_puct_init, &legal_actions)
            };

            let selected_action = next_node_rc.borrow().action;
            //mcts_log!("Selected action: {}", selected_action);

            tree_traversal_path.push(selected_action); // Store the action in the traversal path
            //////////////////////////////////////

            // create the correct chip
            let chip = loop_game_state.current_player().chip_options[0];
            let chip_owner = match loop_game_state.current_player().chip_options[0] {
                RED_CHIP => 1, // Red
                YELLOW_CHIP => -1, // Yellow
                _ => panic!("Invalid player chip option"),
            };
            //mcts_log!("Inserting chip {} at action {}", chip_owner, selected_action);
    
            // Create a new game state
            loop_game_state.play_no_check(selected_action as isize, chip);

            //mcts_log!("Computing board state");
            let board_state = loop_game_state.compute_board_state();
            //mcts_log!("Board state after move: {:?}", board_state);
            
            if board_state != BoardState::Ongoing {
                //mcts_log!("Game over detected, setting over=true");
                over = true;
                break;
            }

            // Update the node reference to the next node
            node_rc = next_node_rc;

            //when loop breaks, loop_game_state is the game state after the last move
        }
        
        // we check why the loop broke 
        if over {
            //mcts_log!("Game is over, computing final value...");
            match loop_game_state.compute_board_state() {
                BoardState::Ongoing => {
                    //mcts_log!("Game state: Ongoing");
                },
                BoardState::Win(side) => {
                    let mut player = match loop_game_state.current_player().chip_options[0] {
                        RED_CHIP => 1, // Red
                        YELLOW_CHIP => 2, // Yellow
                        _ => panic!("Invalid player chip option"),
                    };
                    player = -player; // Since we are checking the player after the move, we need to negate it
                    //mcts_log!("Game state: Win({}), current player: {}", side, player);
                    if player == side as i32 {
                        value = 1.0;
                    } else {
                        value = -1.0;
                    }
                    //mcts_log!("Assigned value: {}", value);
                }
                BoardState::Draw => {
                    //mcts_log!("Game state: Draw");
                    value = 0.0;
                },
                BoardState::Invalid => {
                    //mcts_log!("Game state: Invalid");
                    value = 0.0;
                },
            }
            //mcts_log!("Backpropagating from node: {} value: {}",node_rc.borrow().action, value);
            backpropagate(&root_rc, value, tree_traversal_path);
            //mcts_log!("Backpropagation complete");
        } else {

            let current_player = match loop_game_state.current_player().chip_options[0] {
                RED_CHIP => 1, // Red
                YELLOW_CHIP => -1, // Yellow
                _ => panic!("Invalid player chip option"),
            };
            //mcts_log!("Current player: {}", current_player);

            //mcts_log!("Game not over, calling model_predict...");
            let (prior_probs, value) = model_predict(&loop_game_state, loop_game_history, nnmodel);
            //mcts_log!("Model predict of node {} returned - Value: {}, Prior probs length: {}",node_rc.borrow().action, value, prior_probs.len());
            
            //mcts_log!("Expanding node...");
            expand(&mut node_rc.borrow_mut(), prior_probs, -current_player);
            //mcts_log!("Node expanded successfully");

            //mcts_log!("Backpropagating value: {}", value);
            backpropagate(&root_rc, value, tree_traversal_path);
            //mcts_log!("Backpropagation complete");
        }

        // NO COPYING BACK NEEDED - we're working directly with the original tree
        //mcts_log!("Root visits now: {}", root_rc.borrow().visits);

        let root_borrowed = root_rc.borrow();
        log_tree_visualization(&*root_borrowed, simulation_count, 3);

    }

    //mcts_log!("\n=== MCTS Search Complete ===");
    //mcts_log!("Total simulations performed: {}", simulation_count);
    //mcts_log!("Final root visits: {}", root_rc.borrow().visits);

    // Use root_rc for all final calculations
    let search_policy = if warmup {
        //mcts_log!("Generating search policy with temperature 1.0 (warmup)");
        generate_search_policy(root_rc.borrow().child_visits(), 1.0)
    } else {
        //mcts_log!("Generating search policy with temperature 0.1");
        generate_search_policy(root_rc.borrow().child_visits(), 0.1)
    };

    //mcts_log!("Search policy: {:?}", search_policy);

    let action = match determenistic {
        true => {
            //mcts_log!("Using deterministic action selection");
            let action = search_policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index as u32)
                .expect("No actions available");
            //mcts_log!("Selected action: {}", action);
            action
        }
        false => {
            //mcts_log!("Using stochastic action selection");
            let mut rng = thread_rng();
            let dist = WeightedIndex::new(&search_policy).expect("Invalid distribution");
            let action = dist.sample(&mut rng) as u32;
            //mcts_log!("Sampled action: {}", action);
            action
        }
    };

    //mcts_log!("Final selected action: {}", action);

    let next_root = {
        let root_borrowed = root_rc.borrow();
        root_borrowed.children.get(&action).expect("Action not found in children").clone()
    };
    next_root.borrow_mut().parent = None;

    let best_child_Q = -next_root.borrow().Q_value();
    //mcts_log!("Best child Q-value: {}", best_child_Q);
    //mcts_log!("Root node value: {}", root_rc.borrow().value);

    //mcts_log!("=== MCTS Search Function Complete ===\n");

    return (action, search_policy, best_child_Q, root_rc.borrow().value, next_root.clone());
}