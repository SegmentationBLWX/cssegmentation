'''
Function:
    Implementation of LoggerHandles
Author:
    Zhenchao Jin
'''
import time
from .modulebuilder import BaseModuleBuilder


'''LocalLoggerHandle'''
class LocalLoggerHandle():
    def __init__(self, logfilepath):
        self.logfilepath = logfilepath
    '''log'''
    def log(self, message, level='INFO', endwithnewline=True):
        message = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {level}  {message}'
        print(message)
        if not message.endswith('\n') and endwithnewline:
            message = message + '\n'
        with open(self.logfilepath, 'a') as fp:
            fp.write(message)
    '''debug'''
    def debug(self, message, endwithnewline=True):
        self.log(message, 'DEBUG', endwithnewline)
    '''info'''
    def info(self, message, endwithnewline=True):
        self.log(message, 'INFO', endwithnewline)
    '''warning'''
    def warning(self, message, endwithnewline=True):
        self.log(message, 'WARNING', endwithnewline)
    '''error'''
    def error(self, message, endwithnewline=True):
        self.log(message, 'ERROR', endwithnewline)


'''LoggerHandleBuilder'''
class LoggerHandleBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'LocalLoggerHandle': LocalLoggerHandle,
    }
    '''build'''
    def build(self, logger_handle_cfg):
        return super().build(logger_handle_cfg)


'''BuildLoggerHandle'''
BuildLoggerHandle = LoggerHandleBuilder().build