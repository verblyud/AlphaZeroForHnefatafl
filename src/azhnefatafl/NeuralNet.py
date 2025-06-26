import os
import signal
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pickle
from collections import deque
from typing import Literal
import json
import csv

from .utils import *
from .taflNNet import TaflNNet




from ._azhnefatafl import con4_self_play_function

# args = {
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': torch.cuda.is_available(),
#     'num_channels': 512,
#     'maxlen': 10000,
#     'numGamesPerGen': 100,
#     'mcts': 100,
# }

# game = {
#     'boardsize' : (7, 7),
#     'actionsize' : 49 * 49,
#     }

class NNetWrapper():
    def __init__(self, args, game, jit_model=None):

        name = input("Please specify a name for this wrapper instance: ")
        p = os.path.join('agents', name)
        while os.path.exists(p):
            name = input("The name you chose already exists. Please choose a different name: ")
            p = os.path.join('agents', name)
        self.name = name

        if jit_model is not None:
            self.nnet = jit_model
            print("You created a wrapper instance from a pre-trained model.")
            time.sleep(1)
            if args["cuda"]:
                self.nnet = jit_model.to('cuda')
                print("model was saved to cuda")

        self.args = args
        self.game = game

        self.checkpoint_paths = []
        self.train_examples_paths = []   
        self.gen = 0   
        self.logpath = os.path.join('agents', self.name, 'log.txt')

        print("You chose to create an wrapper instance with the following configuration\n")
        print(vars(self))
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborting...")
            sys.exit()
        
        os.mkdir(p)

        self.log_message("New wrapper was created")
        with open(self.logpath, 'a') as f:
            json.dump(self.__dict__, f, indent = 4)

        
    
    def log_message(self, message):
        timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
        with open(self.logpath, 'a') as f:
            f.write(f"{timestamp} - {message}\n")


    def train(self, examples, summary_writer=None, loss_writer=None):
        """
        examples: list of examples, each example is of form (board, pi, player, v)
        board should already be a matrix representation
        v should always be from the perspective of the attacker 
        """
        optimizer = optim.Adam(self.nnet.parameters())
        if not self.nnet:
            print("this wrapper instance has no attribute nnet. Load an already existing model and try again.")
            return None
        elif self.args['cuda']:
            self.nnet.to('cuda')
        
        for epoch in range(self.args['epochs']):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()

            if loss_writer:
                sum_l_pi = 0
                sum_l_v = 0

            batch_count = int(len(examples) / self.args['batch_size'])

            t = tqdm(range(batch_count), desc='Training Net')
            for step in t:
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])

                # If the examples are given as a structured numpy array. That is, if it was loaded from .npz file
                if isinstance(examples, np.ndarray):
                    boards = torch.FloatTensor(examples["boards"])
                    target_pis = torch.FloatTensor(examples["pis"])
                    players = torch.BoolTensor(examples["players"])
                    target_vs = torch.FloatTensor(examples["vs"])
                else:
                    boards, pis, players, vs = list(zip(*[examples[i] for i in sample_ids]))
                    # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    boards = torch.FloatTensor(np.array(boards))
                    target_pis = torch.FloatTensor(np.array(pis))
                    players = torch.BoolTensor([True if player == 1 else False for player in players])
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.args['cuda']:
                    boards, target_pis, players, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), players.contiguous().cuda(), target_vs.contiguous().cuda()

                pis, vs = self.nnet(boards, players)   
                l_pi = self.loss_pi(target_pis, pis)
                l_v = self.loss_v(target_vs, vs)

                if loss_writer:
                    sum_l_pi += l_pi.item()
                    sum_l_v += l_v.item()

                if summary_writer:
                    summary_writer.add_scalar(f'gen{self.gen}/policy loss', l_pi, step)
                    summary_writer.add_scalar(f'gen{self.gen}/value loss', l_v, step)

                total_loss = l_pi + l_v
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            if loss_writer and batch_count > 0:
                ave_l_pi = sum_l_pi / batch_count
                ave_l_v = sum_l_v / batch_count
                loss_writer.writerow([f'{self.gen}', epoch, ave_l_pi, ave_l_v])

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


    def save_checkpoint(self, prefix: Literal['i','c'], folder='models', filename = None):
        """
        this will save jit.scripted model into a filepath dependent on the current time( e.g. chp_0951_01.02.25.pt)
        & return that filepath as the output.
        prefix 'i' indicates that the model was initialized. 'c' indicates that it is a successor to another checkpoint
        """
        if not filename:
            cur_time = time.strftime("%H%M_%d.%m.%y")
            filename = prefix + '_' + cur_time + '.pt'

        folderpath = os.path.join('agents',self.name, folder)
        filepath = os.path.join(folderpath, filename)

        if not os.path.exists(folderpath):
            print("Checkpoint Directory does not exist! Making directory {} under {}".format(folder, self.name))
            os.mkdir(folderpath)
        
        self.nnet.save(filepath)
        print("scripted model saved at: {}".format(filepath)) 
        self.log_message("scripted model saved at: {}".format(filepath))       
        return filepath

    def load_checkpoint(self, filepath = None):
        if not filepath:
            if self.checkpoint_paths:
                filepath = self.checkpoint_paths[-1]
            else:
                print("You provided no model to load and there is no checkpoint history to this wrapper.")
                sys.exit()

        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        scripted_model = torch.jit.load(filepath)
        self.nnet = scripted_model
        print("model loaded successfully")
        self.log_message("model loaded from {}".format(filepath))
        time.sleep(1)
        if self.args['cuda']:
            self.nnet = scripted_model.to('cuda')
            print("model loaded to cuda")
            time.sleep(1)

    def generate_train_examples(self, model_path, old_train_examples = None, verbose=True):
        if old_train_examples: # Pass a deque object with maxlen specified
            train_examples = old_train_examples
        else:
            train_examples = deque([], maxlen= self.args['maxlen'])
        
        t = tqdm(range(self.args['numGamesPerGen']), desc="generating train examples")
        for _ in t:
            t.set_description(f"Iteration: {_ + 1}\n")
            # Generate new training examples using self-play








        ###############################################
        ############################################### This is where the self-play function for each game is specified
        ###############################################



            logger_path = self.logpath

            new_examples = con4_self_play_function(model_path, 1, self.args['mcts'], verbose, self.args['num_workers'], self.args['c_puct'], self.args['alpha'], self.args['eps'],logger_path)

            # new_examples = con4_self_play_function(model_path, 1, self.args['mcts'], verbose, self.args['mcts_alg'], self.args['num_workers'], self.args['c_puct'], self.args['alpha'], self.args['eps'])
            # add the new_examples to the right side of the deque (if length exceeds the maxlen, it will discard older examples from the left)
            train_examples.extend(new_examples)

        reformatted = []
        for example in train_examples:
            np_matrix = np.array([np.frombuffer(row, dtype=np.uint8) for row in example[0]])
            tmp = (np_matrix,example[1],example[2],example[3])
            reformatted.append(tmp)    
        
        self.log_message("train examples were generated using {}".format(model_path))
        
        return reformatted
    
    def save_train_examples(self, train_examples, folder='train_examples', filename = None):
        if not filename:
            cur_time = time.strftime("%H%M_%d.%m.%y")
            filename = cur_time + '.npz'
        
        folderpath = os.path.join('agents',self.name, folder)
        filepath = os.path.join(folderpath, filename)

        if not os.path.exists(folderpath):
            print("Train examples directory does not exist! Making directory {}".format(folderpath))
            os.mkdir(folderpath)

        dtypes = np.dtype([
            ("boards", np.uint8, self.game['boardsize']),
            ("pis", np.float32, (self.game['actionsize'])),
            ("players", np.int8),
            ("vs", np.float32)
        ])

        np_array = np.array(train_examples, dtype=dtypes)
        np.savez_compressed(filepath, a=np_array)
        print("train_examples saved at {}".format(filepath))
        self.log_message("train examples saved at {}".format(filepath))
        time.sleep(1)
        return filepath

    def load_train_examples(self, path, use: Literal["train", "generate"]):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
        loaded = np.load(path)
        
        if use == "train":
            return loaded['a']
        else:
            return deque((item.item() for item in loaded['a']), maxlen=self.args['maxlen'])

    def learn(self, checkpoint_filepath = None, train_examples_path = None, verbose = True, maxgen = None):

        if checkpoint_filepath:
            user_input = input("You chose to give an external checkpoint path," 
                                + "possibly not related to this wrapper. Are you sure? (y/n)")
            if user_input.lower() != 'y':
                print('aborting...')
                sys.exit()
                    
            self.checkpoint_paths.append(checkpoint_filepath)
            self.log_message(f"An external checkpoint filepath {checkpoint_filepath} was specified when calling the learn function")
        
        elif not self.checkpoint_paths:
            print("No external checkpoint path was given. Let's start fresh! Creating a new model..")
            time.sleep(1)
            initial_model = TaflNNet(self.game, self.args)
            scripted_initial_model = torch.jit.script(initial_model)
            self.nnet = scripted_initial_model
            if self.args['cuda']:
                self.nnet = scripted_initial_model.to('cuda')
            path = self.save_checkpoint(prefix='i', filename= "gen" + f'{self.gen}' + ".pt")
            self.checkpoint_paths.append(path)
            print("New model was created and saved at {}".format(path))
            time.sleep(1)

        if train_examples_path:
            user_input = input("You chose to give an external train example," 
                                + "possibly not related to this wrapper. Are you sure? (y/n)")
            if user_input.lower() != 'y':
                print('aborting...')
                sys.exit()
                    
            self.train_examples_paths.append(train_examples_path)
            old_train_examples = self.load_train_examples(self.train_examples_paths[-1], 'generate')
            self.log_message(f"An external train examples path {train_examples_path} was specified when calling the learn function")

        elif not self.train_examples_paths:
            print("No external train example was given and we are starting fresh!")
            time.sleep(1)
            old_train_examples = deque([], maxlen = self.args['maxlen'])
            path = self.save_train_examples(old_train_examples)
            self.train_examples_paths.append(path)

        def handle_user_exit():
            print("\nManual exit detected. Shutting down the cycle and saving the checkpoint..")
            f.close()
            time.sleep(1)
            print("Latest checkpoints are.. \nmodel:{} \ntraining_examples:{}"
                  .format(self.checkpoint_paths[-1], self.train_examples_paths[-1]))
            time_elapsed = time.time() - start_time
            time_elapsed_minutes = time_elapsed / 60
            self.log_message(f"Time elapsed: {time_elapsed_minutes:.2f} minutes")
            self.save_itself()
            sys.exit()
        
        start_time = time.time()
        # signal.signal(signal.SIGINT, handle_ctrl_c)

        summary_writer = SummaryWriter(log_dir=os.path.join('agents', self.name, 'vcycle'))
        loss_record_path = os.path.join('agents', f'{self.name}', "loss_record.csv")
        f = open(loss_record_path, "a", newline="")
        loss_writer = csv.writer(f)
        loss_writer.writerow(['gen', 'epoch', 'l_pi', 'l_v'])

        try: 
            while True:

                print("Starting the virtuous train cycle! generation: {}".format(self.gen))

                print("Using model at {}".format(self.checkpoint_paths[-1]))

                print("Using the latest train_example at {}".format(self.train_examples_paths[-1]))
                time.sleep(1)
                old_train_examples = self.load_train_examples(self.train_examples_paths[-1], 'generate')
                
                train_examples = self.generate_train_examples(self.checkpoint_paths[-1], old_train_examples, verbose)
                path = self.save_train_examples(train_examples, filename = 'gen' + f'{self.gen}' + '.npz')
                self.train_examples_paths.append(path)
                self.log_message("newer train examples were made using gen {}".format(self.gen))
                self.load_checkpoint(self.checkpoint_paths[-1])

                #Train the model
                self.train(train_examples, summary_writer, loss_writer)
                f.flush()
                self.log_message("gen {} was trained. Entering gen {}..".format(self.gen, self.gen + 1))
                self.gen += 1
                #free up the space
                del train_examples
                path = self.save_checkpoint('c', filename= 'gen' + f'{self.gen}' + '.pt')
                self.checkpoint_paths.append(path)

                if maxgen and self.gen > maxgen:
                    print("\n Generation count reached the user-specified limit. Saving the checkpoint..")
                    f.close()
                    time.sleep(1)
                    print("Latest checkpoints are.. \nmodel:{} \ntraining_examples:{}"
                        .format(self.checkpoint_paths[-1], self.train_examples_paths[-1]))
                    time_elapsed = time.time() - start_time
                    time_elapsed_minutes = time_elapsed / 60
                    self.log_message(f"Time elapsed: {time_elapsed_minutes:.2f} minutes")
                    self.save_itself()
                    sys.exit()
        
        except KeyboardInterrupt:
            handle_user_exit()


            
    def save_itself(self, savennmodelaswell = False):
        if not savennmodelaswell:
            self.nnet = None

        self.log_message("Wrapper saved.")
        with open(self.logpath, 'a') as f:
            json.dump(self.__dict__, f, indent=4)
        
        filepath = os.path.join('agents', self.name, 'wrapper.pkl')
        if os.path.exists(filepath):
            print('overwriting..')

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print("wrapper saved! To load, use load_wrapper function")
    
    def change_arg(self, argname: Literal['lr','dropout','epochs','batch_size','cuda','num_channels','maxlen','numGamesPerGen', 'mcts'], new):
        if not isinstance(argname, str):
            print("Please provide a string")
            return
        
        if argname not in self.args:
            print(f"You tried to change a non-existent argument: {argname}")
            return
        
        old_value = self.args[argname]
        if not isinstance(new, type(old_value)):
            print(f"Type mismatch: {argname} should be of type {type(old_value).__name__}")
            return

        self.log_message(f"The argument {argname} was changed from {old_value} to {new}")
        self.args[argname] = new
        
def load_wrapper(wrapper_name) -> NNetWrapper:
    filepath = os.path.join('agents', wrapper_name, 'wrapper.pkl')
    if not os.path.exists(filepath):
        print("Could not find the wrapper you specified..")
        sys.exit()
    with open(filepath, 'rb') as f:
        loaded_wrapper = pickle.load(f)
    print("The loaded wrapper has the following attributes..")
    print(vars(loaded_wrapper))
    loaded_wrapper.load_checkpoint()
    loaded_wrapper.log_message("wrapper was loaded")
    return loaded_wrapper   
     



