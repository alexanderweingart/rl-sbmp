import datetime
import git
class RepoChecker:
    IGNORE_DICTS = [
        "thesis",
        "logs"
    ]

    def __init__(self):
        self.repo = git.Repo(search_parent_directories=True)

    def is_dirty(self, verbose: bool = True):
        dirty_files = [item.a_path for item in self.repo.index.diff(None)
                       if item.a_path.split("/")[0] not in self.IGNORE_DICTS]
        if len(dirty_files) > 0:
            if verbose:
                print("[red] >>> The repo is dirty (uncommitted files) <<<")
                print(f"{dirty_files}")
            return True
        else:
            print("[red] >>> The repo is clean <<<")
            return False

    def get_commit_id_hash(self):
        return self.repo.head.object.hexsha

    def get_commit_dir_name(self):
        sha = self.repo.head.object.hexsha
        unix_time = self.repo.head.object.committed_date
        time_stamp = datetime.datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d_%H-%M-%S")
        return f"{time_stamp}_{sha}"
