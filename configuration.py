import sys
import json
#import Augmentor

AUGMENT_CONFIG = {
     "Path" : "~/udacity/source",
     "Game_Name" : "CartPole-v0",
     "BATCH_SIZE" : "128", 
     "GAMMA" : "0.999",
     "EPS_START" : "0.9",	 
     "EPS_END" : "0.05",	 
     "EPS_DECAY" : "200",	 
     "TARGET_UPDATE" : "10",	 
     "Num_Episodes" : "50",	 	
     "ReplayMemory_Size" : "10000"	 	 
}

def displayconfig(config):
    # List all contents
    print("List all name and value")
    for section in config:
        print("Section: %s" % section)
        for options in config[section]:
            print("    %s=%s" % (options, config[section][options]))


def loadConfig(config_file_name):
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    return config
 
		
	
def augment():	
    p = Augmentor.Pipeline(AUGMENT_CONFIG["Image_path"])
    p.rotate90(float(AUGMENT_CONFIG["rotate90_probability"]))
    p.rotate270(float(AUGMENT_CONFIG["rotate270_probability"]))
    p.flip_left_right(float(AUGMENT_CONFIG["flip_left_right_probability"]))
    p.flip_top_bottom(float(AUGMENT_CONFIG["flip_top_bottom_probability"]))
    p.crop_random(float(AUGMENT_CONFIG["crop_random_probability"]), float(AUGMENT_CONFIG["crop_random_percentage_area"]))
    p.resize(float(AUGMENT_CONFIG["resize_probability"]), int(AUGMENT_CONFIG["resize_width"]), int(AUGMENT_CONFIG["resize_height"]))
    p.sample(int(AUGMENT_CONFIG["sample_number"]))

if __name__ == "__main__":
    config_file_name = sys.argv[1]
    print("config file name = ", config_file_name)
    try:
        config = loadConfig(config_file_name)
#        print(config)
        displayconfig(config)
    except:
        print('message: ', sys.exc_info())   	