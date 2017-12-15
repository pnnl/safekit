from datetime import datetime


class Merge:
    """
    Live merging of csv files. The call of this object is a generator function
    which interleaves lines from a collection of files, ordered by a sort_column
    parameter.

    Assumes:
        (i) Individual files are ordered by ascending sort column values.
        (ii) Individual files have headers with one column named the same as <sort_column> parameter.
        (iii) Files to merge are in the same folder specified by <file_path> parameter>.

    The generator operates as follows:
        (i) Upon initialization, aligned lists of files, file names, file headers, and
            the first non-header line (split on delimiter with file-type index appended)
            of each file are constructed.
        (ii) When the Merge object is called the list of lines is sorted by time-stamp specified by <sort_column>
             and <date_format> parameters.
        (iii) The line (split on delimiter) with the earliest time stamp is returned along with
              the name of the file it came from (determined by appended event_type int).
        (iv) The line is replaced from the file it came from (determined by appended event_type int).
        (v) If there are no more lines left in the file then it is closed and list entries associated with this file are
            removed from lists (determined by appended event_type int).
        (vi) Concludes generating when all files are ended.
    """

    def __init__(self, filepath='./',
                 file_list=['short_t_toy_auth.txt', 'short_t_toy_proc.txt'],
                 sort_column='time',
                 date_format='int',
                 delimiter=','):
        """

        :param filepath: Path to folder with files to merge.
        :param file_list: List of names of files to merge.
        :param sort_column: Column to sort lines of files on for sequential ordering of log lines.
        :param date_format: Can be any format string which makes sense to datetime.strptime or 'int' for simple integer time stamps.
        :param delimiter: Delimiter of csv columns, e.g. ',', ' ' ...
        """
        if not filepath.endswith('/'):
            filepath += '/'
        self.file_list = file_list
        self.filepath = filepath
        self.files = [open(filepath + f, 'r') for f in file_list]
        self._headers = [f.readline().strip().split(',') for f in self.files]
        self.sorters = [header.index(sort_column) for header in self._headers]
        self.event_types = range(len(self.files))
        self.events = [f.readline().strip().split(delimiter) + [idx] for idx, f in enumerate(self.files)]
        self.event_lengths = [len(header) for header in self._headers]
        self.auth_index = 0
        self.proc_index = 0
        if date_format == 'int':
            self.sort_function = lambda x: int(x[self.sorters[x[-1]]])
        else:
            self.sort_function = lambda x: datetime.strptime(x[self.sorters[x[-1]]], self.date_format)

    @property
    def headers(self):
        """
        :return: A list of headers (split by delimiter) from files being merged
        """
        return self._headers

    def next_event(self, event_type):
        """
        :param event_type: Integer associated with a file to read from.
        :return: Next event (line from file split on delimiter with type appended) from file associated with event_type.

        """
        return self.files[event_type].readline().strip().split(',') + [event_type]

    def __call__(self):
        """
        :return: (tuple) filename, Next event (line from file split on delimiter) according to time stamp.
        """
        while True:
            # try:
            if len(self.events) == 0:
                return
            elif len(self.events) == 1:
                if self.events[0][0] == '':
                    return
                # else:
                #     event_type = self.events[0][:-1]
                #     least_time_event = self.events[0][:-1]
                #     self.events[0] = self.next_event(event_type)
                #     yield self.file_list[event_type], least_time_event

            least_time_event = sorted(self.events, key=self.sort_function)[0]
            event_type = least_time_event[-1]
            yield self.file_list[event_type], least_time_event[:-1]
            new_event = self.next_event(event_type)
            if new_event[0] == '':
                self.files[event_type].close()
                self.files.pop(event_type)
                self.sorters.pop(event_type)
                self.event_types.pop(event_type)
                self.events.pop(event_type)
                self.file_list.pop(event_type)
                self.event_lengths.pop(event_type)
                if len(self.files) == 0:
                    return
            else:
                self.events[event_type] = new_event
                assert len(self.events[event_type]) == self.event_lengths[event_type] + 1


