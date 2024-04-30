import sys
 
'''/////////////////User input//////////////////////////////////'''
def checkInput(mess):
     while True:
            try:
                inp = input(mess)
                i = int(inp)
            except ValueError:
                print("Please enter an integer!")
                continue
            else:
                if i == -1:
                    sys.exit()
                
                return i        


def checkTrim(signal, fs):
    inp = input("Trim signal?")
    #Not finished
        
def yes(inp):
    if inp == 'y' or inp == 'Y':
        return True
    return False
def get_trial(numTrials):         
    while True:
        while True:
            try:
                inp = input("Select trial number [0," + str(numTrials-1) + "]: ")
                i = int(inp)
            except ValueError:
                print("Please enter an integer!")
                continue
            else:
                break
        
        if i < 0:
            print("Program exiting...")
            sys.exit()
        elif i >= numTrials:
            print("Trial number too large. Trial number must be less than " + str (numTrials))
        else:
            return i
