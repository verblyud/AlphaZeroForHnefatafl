

fn con4_search_par( 
    game_state: Game, 
    root_node: &mut Node, 
    nnmodel: &CModule, 
    // Parameters
    c_puct_base: f32,
    c_puct_init: f32, 
    num_simulations: u32, 
    num_parallel: i32,
    root_noise: bool, 
    warmup: bool, 
    determenistic: bool,
    logger_path: &str) -> (u32, Vec<f32>, f32, f32, Rc<RefCell<Node>>) {
    

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

    let root_rc = Rc::new(RefCell::new(std::mem::take(root_node)));


    let mut simulation_count = 0;

    negate_expanded_values(&root_rc);

    while root_rc.borrow().visits < num_simulations as f32 + num_parallel as f32 {

        //let mut leaves = Vec::new();
        let mut failsafe = 0;



        while leaves.len() < num_parallel as usize && failsafe < num_parallel * 2 {

            failsafe +=1;
            let mut tree_traversal_path = Vec::new();

        
            simulation_count += 1;
            //mcts_log!("\n--- Simulation {} ---", simulation_count);
            //mcts_log!("Root visits: {}/{}", root_rc.borrow().visits, num_simulations);

            // Safety check for infinite loop
            if simulation_count as u32 > num_simulations * 2 {
                //mcts_log!("ERROR: Too many simulations! Breaking to prevent infinite loop.");
                break;
            }

            let mut node_rc = root_rc.clone(); // This creates a reference to the node that will be used in the simulation

            let mut value: f32 = 0.0;
            let mut over = false;
            let mut loop_game_state = game_state.clone();
            let mut traversal_depth = 0;


            loop {

                traversal_depth += 1;
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

            add_virtual_loss(&next_node_rc, tree_traversal_path);

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

            }

        }

    }


    // This function is a parallel version of con4_search
    // It will use the same logic but will run multiple simulations in parallel
    // using the rayon crate

    unimplemented!();
}


fn con4_search_root_par (
    game_state: Game, 
    root_node: &mut Node, 
    nnmodel: &CModule, 
    // Parameters
    c_puct_base: f32,
    c_puct_init: f32, 
    num_simulations: u32, 
    num_parallel: i32,
    root_noise: bool, 
    warmup: bool, 
    determenistic: bool,
    logger_path: &str) -> (u32, Vec<f32>, f32, f32, Rc<RefCell<Node>>) {

    
    // This function is a parallel version of con4_search
    // This implementation will run several shallow trees in parallel, unsure if this is best
    // try to implement the github method with vl_loss instead

    let shared_model = Arc::new(nnmodel);
    let pool = ThreadPool::new(num_parallel);
    let (tx, rx) = mpsc::channel();
    let tx = Arc::new(Mutex::new(tx));
    let num_iter_per_worker: usize = num_simulations as usize / num_parallels as usize;
    for _ in 0..num_parallel as usize {
        let tx = Arc::clone(&tx);
        let nnmodel = Arc::clone(&shared_model);
        let game_state = game_state.clone();

        pool.execute(move || {
            let (action, search_policy, best_child_Q, root_value, next_root) = con4_search(
                game_state,
                root_node,
                &nnmodel,
                c_puct_base,
                c_puct_init,
                num_iter_per_worker as u32,
                root_noise,
                warmup,
                determenistic,
                logger_path
            );
            tx.lock().unwrap().send((action, search_policy, best_child_Q, root_value, next_root));
        });
        let mut final_policy = vec![0.0; 7_usize.pow(4)];
    for policy in rx.iter().take(num_workers) {
        for (i, p) in policy.iter().enumerate() {
            final_policy[i] += p;
        }
    }

    for p in &mut final_policy {
        *p /= num_workers as f32;
    }
    final_policy


    }
}



fn add_virtual_loss(root_rc: &Rc<RefCell<Node>>,tree_traversal_path: Vec<u32>) -> () {

    let vloss = 1.0; // Virtual loss value to be added

    if tree_traversal_path.is_empty() {
        //mcts_log!("No actions in traversal path, updating root.");
        root_rc.borrow_mut().v_loss += vloss; // Update root value directly
        root_rc.borrow_mut().vl_applied += 1; // Increment root visits
        return;
    }

    root_rc.borrow_mut().v_loss += vloss; // Update root value directly
    root_rc.borrow_mut().vl_applied += 1; // Increment root visits

    let mut update_rc = Rc::clone(root_rc);

    //mcts_log!("Virtual loss added. New value: {}, visits: {}", node.value, node.visits);
    for action in tree_traversal_path.iter() {
        
        // Get the child node
        let child_rc = {
            let node_borrowed = update_rc.borrow();
            node_borrowed.children.get(action)
                .expect("Action not found in children")
                .clone()
        };
        
        // Update the child's value and visits
        root_rc.borrow_mut().v_loss += vloss; // Update root value directly
        root_rc.borrow_mut().vl_applied += 1; // Increment root visits

        // Move to the child node for next iteration
        update_rc = child_rc;
        
    }

}


fn revert_virtual_loss(root_rc: &Rc<RefCell<Node>>, tree_traversal_path: Vec<u32>) -> () {

    let vloss = 1.0; // Virtual loss value to be reverted

    if tree_traversal_path.is_empty() {
        //mcts_log!("No actions in traversal path, updating root.");
        root_rc.borrow_mut().v_loss -= vloss; // Update root value directly
        root_rc.borrow_mut().vl_applied -= 1; // Increment root visits
        return;
    }

    root_rc.borrow_mut().v_loss -= vloss; // Update root value directly
    root_rc.borrow_mut().vl_applied -= 1; // Increment root visits

    let mut update_rc = Rc::clone(root_rc);

    for action in tree_traversal_path.iter() {
        
        // Get the child node
        let child_rc = {
            let node_borrowed = update_rc.borrow();
            node_borrowed.children.get(action)
                .expect("Action not found in children")
                .clone()
        };
        
        // Update the child's value and visits
        child_rc.borrow_mut().v_loss -= vloss; // Update root value directly
        child_rc.borrow_mut().vl_applied -= 1; // Increment root visits

        // Move to the child node for next iteration
        update_rc = child_rc;
        
    }

}