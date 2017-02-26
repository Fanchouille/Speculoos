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
    modelsPath = iHomePath + '/Models/'
    resultsPath = iHomePath + '/Results/'
    pfPath = iHomePath + '/Portfolio/'
    Paths = {'DataPath': dataPath, 'LogsPath': logsPath, 'ModelsPath': modelsPath, 'ResultsPath': resultsPath,
             'PFPath': pfPath}
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
