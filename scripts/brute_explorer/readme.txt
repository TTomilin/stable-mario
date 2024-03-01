## Guide to brute explorer script:
	* Arguments:
		- First argument is name of state to load. This state must be located
		in the MarioParty-GbAdvance folder that is found in the stable-retro
		directory in your conda environment. It is the state that the random
		agent will load and then play from. Make sure it is the most up-to-date
		state, because all discovered minigames will be added to this script.
		
		- Second argument is the name of the file to which the script will
		save the state upon keyboard interrupt. It can be anything you like,
		as long as it ends with '.state'.
	
	* Description:
		Loads specified state, hits random buttons until you press 'ctrl+C'
		on the terminal, saves state to specified file in current directory.