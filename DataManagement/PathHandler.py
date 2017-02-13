import os


def create_path(iPath):
    """

    :param iPath: path of dir to be created if doesn't exist
    :return:
    """
    if not os.path.exists(iPath):
        os.makedirs(iPath)
    return


def get_all_paths(iHomePath):
    """

    :param iHomePath:
    :return: Paths list
    """
    dataPath = iHomePath + '/Data/'
    logsPath = iHomePath + '/Logs/'
    Paths = {'DataPath': dataPath, 'LogsPath' : logsPath}
    return Paths


def create_all_paths(iHomePath):
    """

    :param iHomePath: homepath of project
    :param iName: create all path of generic
    :return:
    """
    Paths = get_all_paths(iHomePath)
    for path in Paths.keys():
        create_path(Paths[path])
    return
