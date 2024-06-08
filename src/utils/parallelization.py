import psutil


def get_current_childs() -> set:
    """
    Get the current child processes of the current process.

    Returns
    -------
    set
        Set of the PIDs of the current child processes.
    """
    current_process = psutil.Process()
    return set([p.pid for p in current_process.children(recursive=True)])


def kill_diff_childs(orig_set) -> bool:
    """
    Kill the child processes that are not in the original set of PIDs.

    Parameters
    ----------
    orig_set : set
        Set of PIDs of the original child processes.

    Returns
    -------
    bool
        True if there was an error killing a process, False otherwise.
    """
    error = False
    current_set = get_current_childs()

    for pid in current_set - orig_set:
        try:
            psutil.Process(pid).terminate()
        except psutil.NoSuchProcess:
            error = True
            pass

    return error
