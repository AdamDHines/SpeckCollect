#MIT License

#Copyright (c) 2024 Adam Hines

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''

import argparse

from speckcollect.create_images import Frames
from speckcollect.dvs_collector import Collector

def init(args):
    # Initialize the collector module
    collector = Collector(args.directory,args.exp,time_int=args.time_int)
    # Start the DVS collector
    collector.start_visualizer()
    collector.save_events()
    # Create frames from the events
    framer = Frames(args.directory,args.exp)
    framer.create_frames()

def parse_network():

    # Create the parser
    parser = argparse.ArgumentParser(description='Args for base config file.')

    # Processing arguments
    parser.add_argument('--time_int', type=int, default=1.0,
                    help="Time to collect spikes over for each frame")
    
    # Experimental arguments
    parser.add_argument('--directory', type=str, default='./speckcollect/data',
                    help="Directory to save data")
    parser.add_argument('--exp', type=str, default='exp',
                    help="Experiment name")
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the initialization function
    init(args)

if __name__ == '__main__':
    parse_network()