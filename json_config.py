import sys
import json

# Display a loaded json configuration object
def displayconfig(config):
    # List all contents
    print("List all name and value")
    for section in config:
        print("Section: %s" % section)
        for options in config[section]:
            print("    %s=%s" % (options, config[section][options]))


# Load a configuration object from json format file
def loadConfig(config_file_name):
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    return config
 
		
# Create a dictionary object from a configuration object from json format file	
def create_dictionary(config):	
    dict = {}
    for section in config:
 #       print("Section: %s" % section)
        for options in config[section]:
            prefix = section.strip()
            if len(prefix) > 5:
                prefix = prefix[0:5]
#            print("    %s=%s" % (options, config[section][options]))
            dict[prefix + '_' + options] = config[section][options]
    return dict


# Display all items in a dictionary object
def display_dictionary(dict):
    for k, v in dict.items():
        print("%s=%s" % (k, v))


# Main test code
if __name__ == "__main__":
    config_file_name = sys.argv[1]
    print("config file name = ", config_file_name)
    try:
        config = loadConfig(config_file_name)
#        print(config)
#        displayconfig(config)
        paramters = create_dictionary(config)
        display_dictionary(paramters)
    except:
        print('message: ', sys.exc_info())   	