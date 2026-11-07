"""Entry point mounted alone alongside the historical EvalPlus worker in a sandbox."""
import argparse
import json
import os
import resource
import runpy
import socket


def probe():
    result = {'host_home_hidden': not os.path.exists('/home/joaof'),
              'windows_hidden': not os.path.exists('/mnt/c'),
              'no_credentials': not any('TOKEN' in k or 'SECRET' in k or 'KEY' in k for k in os.environ),
              'address_space_limited': resource.getrlimit(resource.RLIMIT_AS)[0] == 3221225472,
              'cpu_limited': resource.getrlimit(resource.RLIMIT_CPU)[0] == 7200}
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=.5)
        result['network_blocked'] = False
    except OSError:
        result['network_blocked'] = True
    try:
        with open('/app/forbidden', 'w'):
            pass
        result['application_readonly'] = False
    except OSError:
        result['application_readonly'] = True
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['humaneval', 'mbpp'], required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        probe()
    else:
        # run_path avoids adding the repository or the host home to sys.path.
        worker = runpy.run_path('/app/evalplus_worker.py', run_name='isolated_evalplus')
        worker['serve'](args.benchmark)
