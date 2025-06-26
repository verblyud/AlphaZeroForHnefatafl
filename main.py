import sys
sys.path.append("./src")
import azhnefatafl as azh
import os

if azh.Args['cuda']:
    print("CUDA is available.")
else:
    print("CUDA was not found")

def ask_verbose():
    ui = input("Do you want to see the boards and the moves agents make during self play? (y/n)")
    verbose = True if ui.lower() == 'y' else False
    return verbose

def ask_maxgen():
    ui = input("Do you want to set a limit on the maximum count of the generation that will be generated?(y/n)")
    while True:
        if ui.lower() == 'n':
            maxgen = None
            break
        elif ui.lower() == 'y':
            ui2 = input("Specify the maximum count of generations you want to train")
            while True:
                try:
                    n = int(ui2)
                    if n > 0:
                        maxgen = n
                        break
                    else:
                        ui2 = input("Please enter a positive integer for the number of generations:")
                except ValueError:
                    ui2 = input("Please enter a valid integer for the number of generations:")
            break
        else:
            ui = input("please choose y or n")
    return maxgen

def func1():
    verbose = ask_verbose()
    maxgen = ask_maxgen()
    wrapper = azh.NNetWrapper(azh.Args, azh.Game)
    wrapper.learn(verbose=verbose, maxgen = maxgen)

def func2():
    verbose = ask_verbose()
    maxgen = ask_maxgen()

    ui = input("Please type the name of the agent you want to load")
    while True:
        agent_path = os.path.join("agents", ui)
        if os.path.exists(agent_path):
            agentname = ui
            break
        else:
            ui = input("The specified agent does not exist. Please type the name of an existing agent you want to load")
        
    wrapper = azh.load_wrapper(agentname)
    wrapper.learn(verbose=verbose, maxgen=maxgen)





# Main function to start the program
user_input = input("Welcome! If you want to train a model from a scratch, press 1.\n If you want to pick an agent and continue from the last checkpoint, press 2.")


while True:
    if user_input == '1':
        func1()
    elif user_input == '2':
        func2()
    else:
        user_input = input("Please choose 1 or 2")

   

