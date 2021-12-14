import argparse
import json

PARSE = argparse.ArgumentParser()


def get_parse_from_json(json_path='option/args.json'):
    try:
        with open(json_path) as f:
            json_dict = json.load(f)

        args = list()
        for kind in json_dict.keys():
            if kind == 'debug':
                args.append((kind, json_dict[kind]))
            else:
                for k, v in json_dict[kind].items():
                    args.append((k, v))
        for name, params in args:
            action, t, default, h, choices = None, None, None, None, None
            if 'action' in params:
                action = params['action']
            if 'type' in params:
                if params['type'] == 'int':
                    t = int
                elif params['type'] == 'float':
                    t = float
                elif params['type'] == 'str':
                    t = str
                elif params['type'] == 'bool':
                    t = bool
            if 'default' in params:
                default = params['default']
            if 'help' in params:
                h = params['help']
            if 'choices' in params:
                choices = tuple(params['choices'].split())

            if default == 'True':
                default = True
            elif default == 'False':
                default = False

            if choices is not None:
                PARSE.add_argument('--' + name, type=t, default=default, choices=choices, help=h)
            elif action is not None:
                PARSE.add_argument('--' + name, action=action, help=h)
            elif type is not None:
                PARSE.add_argument('--' + name, type=t, default=default, help=h)
        p = PARSE.parse_args()
        if p.scale_1 == '' or p.scale_2 == '':
            if p.asymm:
                p.scale_1 = [
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                    1.5, 1.5, 1.5, 1.5, 1.5,
                    2.0, 2.0, 2.0, 2.0, 2.0,
                    2.5, 2.5, 2.5, 2.5, 2.5,
                    3.0, 3.0, 3.0, 3.0, 3.0,
                    3.5, 3.5, 3.5, 3.5, 3.5,
                    4.0, 4.0, 4.0, 4.0, 4.0,
                ]
                p.scale_2 = [
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                    2.0, 2.5, 3.0, 3.5, 4.0,
                    1.5, 2.5, 3.0, 3.5, 4.0,
                    1.5, 2.0, 3.0, 3.5, 4.0,
                    1.5, 2.0, 2.5, 3.5, 4.0,
                    1.5, 2.0, 2.5, 3.0, 4.0,
                    1.5, 2.0, 2.5, 3.0, 3.5,
                ]
            # symmetric mode: only non-integer scale factors
        return p
    except IOError as e:
        print(e)
