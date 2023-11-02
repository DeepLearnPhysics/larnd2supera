#!/usr/bin/python3
import larnd2supera
import sys,os

from optparse import OptionParser
...
parser = OptionParser(usage='Provide option flags followed by a list of input files.')
parser.add_option("-o", "--output", dest="output_filename", metavar="FILE",
    help="output (LArCV) filename")
parser.add_option("-c", "--config", dest="config", metavar='FILE/KEYWORD', default='',
    help="configuration keyword or a file path (full or relative including the file name)")
parser.add_option("-n", "--num_events", dest="num_events", metavar="INT", default=-1,
    help="number of events to process")
parser.add_option("-s", "--skip", dest="skip", metavar="INT", default=0,
    help="number of first events to skip")
parser.add_option("-b", action="store_true", dest="ignore_bad_association", default=False)
parser.add_option("-l", "--log", dest="log_file", metavar="FILE", default='',
    help="the name of a log file to be created. ")

(data, args) = parser.parse_args()

if os.path.isfile(data.output_filename):
    print('Ouput file already exists:',data.output_filename)
    print('Exiting')
    sys.exit(1)

if not data.config in larnd2supera.config.list_config() and not os.path.isfile(data.config):
    print('Invalid configuration given:',data.config)
    print('The argument is not valid as a file path nor matched with any of')
    print('predefined config keys:',larnd2supera.config.list_config())
    print('Exiting')
    sys.exit(2)


if len(args) < 1:
    print('No input files given! Exiting')
    sys.exit(3)

output = sys.argv[1]
input_files = sys.argv[2:]

if os.path.isfile(sys.argv[1]):
    print('Output file already exists:',output)
    sys.exit(2)

if not data.config:
    print('Configuration file/keyword is required.')
    sys.exit(3)

if not larnd2supera.config.get_config(data.config):
    print('Invalid configuration option argument:',data.config)
    sys.exit(3)

larnd2supera.utils.run_supera(out_file=data.output_filename,
    in_file=args[0],
    config_key=data.config,
    num_events=int(data.num_events),
    num_skip=int(data.skip),
    ignore_bad_association=bool(data.ignore_bad_association),
    save_log=data.log_file,
    )
