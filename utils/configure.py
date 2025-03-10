import argparse
import configparser
import datetime
import os

DEFAULT_CONFIGURE_DIR = "configure"
DEFAULT_DATASET_DIR = "data"
DEFAULT_MODEL_DIR = "models"


class Args(object):
    def __init__(self, contain=None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name[-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if not (value is None) or (name[:2] == "__" and name[-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        # give args elements dictionary order to ensure its replicate-ability
        return sorted(list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__)).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)


class String(object):
    @staticmethod
    def to_basic(string):
        """
        Convert the String to what it really means.
        For example, "true" --> True as a bool value
        :param string:
        :return:
        """
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except ValueError:
                pass
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False
        else:
            return string.strip("\"'")  # for those we want to add space before and after the string


class Configure(object):
    @staticmethod
    def get_file_cfg(file):
        """
        get configurations in file.
        :param file:
        :return: configure args
        """
        cfgargs = Args()
        cfgargs.cfg_path = file
        parser = configparser.ConfigParser()
        parser.read(file)
        for section in parser.sections():
            setattr(cfgargs, section, Args())
            for item in parser.items(section):
                setattr(getattr(cfgargs, section), item[0], String.to_basic(item[1]))
        return cfgargs

    @staticmethod
    def refresh_args_by_file_cfg(file, prev_args):
        args = Configure.get_file_cfg(file)
        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        for arg_name, arg in prev_args:
            if arg is None:
                continue
            if arg_name != "cfg":
                names = arg_name.split(".")
                cur = args
                for name in names[: -1]:
                    if getattr(cur, name) is None:
                        setattr(cur, name, Args())
                    cur = getattr(cur, name)
                if getattr(cur, names[-1]) is None:
                    setattr(cur, names[-1], arg)
        return args


    @staticmethod
    def Get(cfg):
        args = Configure.get_file_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, cfg))

        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        return args

    @staticmethod
    def Get_from_file(cfg):
        args = Configure.get_file_cfg(cfg)

        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        return args

    @staticmethod
    def save_to_file(cfg, file_path):
        """
        静态方法：将 Args 对象保存到文件
        :param cfg: Args 对象
        :param file_path: str, 目标文件路径
        """
        if not isinstance(cfg, Args):
            raise ValueError("配置对象必须是 Args 类型")

        parser = configparser.ConfigParser()

        # 遍历配置对象
        for section in dir(cfg):
            if section.startswith("__"):  # 跳过特殊属性
                continue
            section_obj = getattr(cfg, section)
            if isinstance(section_obj, Args):
                parser[section] = {}
                for item_name, item_value in section_obj:
                    parser[section][item_name] = str(item_value)

        # 写入到文件
        with open(file_path, "w") as file:
            parser.write(file)
        print(f"配置已保存到文件: {file_path}")