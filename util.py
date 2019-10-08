# Helper functions to save a load data.

import sys
import numpy as np
import scipy.misc

def parse_otu_table(filename, sep=",", has_rownames=True):
    """Reads an OTU table of multiple time-series observations. Each
    row is an OTU, and each column is an observation from a single
    time point.
    
    Specifically, the format of the OTU table is:
        Row 1: Unique identifiers per sequence. For example, if
               there are 30 observed sequences, these could be labeled
               from 1, 2,.., 30.
        Row 2: Integers specifying the time-point of each observation.
               For example, if observations are in days, one row could
               look like: 3  5  10 ... .
        Row N: The remaining rows are counts of each OTU across all
               observed sequences and time points.
    
    Together, rows 1 and 2 uniquely identify a time point. For instance,
    the column corresponding to observed sequence 7 at time point 3 should
    have 7 in the first row and 3 in the second row.

    Parameters
    ----------
        filename     : a string denoting the filepath of the OTU table.
        sep          : character denoting separator between observations.
        has_rownames : if True, the first column is ignored

    Returns
    -------
        seq_id       : a numpy array of ids identifying each sequence in the
                       otu_table
        otu_table    : a numpy array of observations. The format is the same
                       as the original otu table
    """

    with open(filename, "r") as f:
        lines = f.readlines()
        n_row = len(lines[1:])

        if has_rownames:
            n_col = len(lines[0].split(sep)) - 1
            seq_id = np.array(lines[0].strip("\n").split(sep))[1:]
        else:
            n_col = len(lines[0].split(sep))
            seq_id = np.array(lines[0].strip("\n").split(sep))

        table = np.zeros((n_row, n_col))

        for idx, line in enumerate(lines[1:]):
            if has_rownames:
                line = line.strip("\n").split(sep)[1:]
            else:
                line = line.strip("\n").split(sep)
            n_col_line = len(line)

            if n_col_line != n_col:
                print_error_and_die("Error parsing OTU table: row " + str(idx) + \
                                    " has " + str(n_col_line) + " columns, but row 0 has" + \
                                    n_col + " columns.")

            try:
                row = np.array(line, dtype=float)
            except ValueError:
                print_error_and_die("Error parsing OTU table: invalid character in row " + str(idx))

            if idx > 1 and np.any(row < 0.):
                print_error_and_die("Error parsing OTU table: row " + str(idx) + " has negative entry.")

            if row.sum() == 0:
                print("\tWarning: row " + str(idx) + " has no nonzero entries.", file=sys.stderr)

            table[idx] = row

        return seq_id, table


def parse_event_table(filename, sep=",", has_header=True):
    """Reads a table of external events; for example antibiotic
       administration over a number of days.
    
    The format of the event table is assumed to have four columns:
        Column 1: Unique ID of the corresponding time series observation
        Column 2: Integer ID corresponding to a particular event.
        Column 3: Time point corresponding to the start of the event. This
                  must be greater than or equal to the time point of the first
                  time series observation.
        Column 4: Time point corresponding to the end of the event. This should be
                  less than or equal to the last time series observation.

    Parameters
    ----------
        sep       : delimiter used to denote separation between columns
        has_header: if True, the first row is ignored.

    Returns
    -------
        events       : a numpy array of (event_integer_id, start_day, end_day)
                       for each event in filename
        event_to_int : a dictionary that maps event names to unique integers
        seq_id       : sequence id of each row in events
    """

    with open(filename, "r") as f:
        lines = f.readlines()
        if has_header:
            lines = lines[1:]

        n_col = len(lines[0].strip("\n").split(sep))
        n_row = len(lines)

        event_types = []
        # parse to find all event types and check formatting
        for idx, line in enumerate(lines):
            if has_header:
                row_num = idx + 1
            else:
                row_num = idx

            line = line.strip("\n").split(sep)
            if len(line) != n_col:
                print_error_and_die("Error parsing event table: row " + str(row_num) + " has " + \
                                    str(len(line)) + " columns, but header has " + str(n_col) + ".")

            try:
                start_day = int(line[2])
                end_day = int(line[3])
            except ValueError:
                print_error_and_die("Error parsing event table: row " + str(row_num) + " has an invalid entry for start day or end day.")

            if line[1] not in event_types:
                event_types.append(line[1])

        # store events in numpy array
        event_to_int = dict( [(event, i) for i, event in enumerate(event_types) ])
        seq_id = []
        events = np.zeros((n_row, 3), dtype=int)

        for idx, line in enumerate(lines):
            line = line.strip("\n").split(sep)
            seq_id.append(line[0])
            e_idx = event_to_int[line[1]]
            start_day = int(line[2])
            end_day = int(line[3])
            row = np.array((e_idx, start_day, end_day), dtype=int)
            events[idx] = row

        print("\tDone!")
        return events, event_to_int, seq_id


def format_observations(otu_table,
                        otu_seq_id,
                        event_table,
                        event_seq_id,
                        event_to_int):
    # counts
    Y = []
    # effects
    U = []
    # times
    T = []

    U_normalized = []
    T_normalized = []

    sort_by_seq = np.argsort(otu_seq_id)
    otu_table = otu_table[:,sort_by_seq]
    otu_seq_id = otu_seq_id[sort_by_seq]
    for s_id in np.unique(otu_seq_id):
        seq = otu_table[:,np.array(otu_seq_id) == s_id]
        sort_by_day = np.argsort(seq[0])
        seq = seq[:,sort_by_day]
        days = np.array(seq[0], dtype=int)
        
        observations = []
        for day in days:
            idx = np.argwhere(days == day)
            idx = idx.reshape(idx.size)
            row = seq[1:,idx]
            if (row.shape[1] > 1):
                print("\tWarning: sequence id " + s_id + " has multiple observations for time point " + str(day), file=sys.stderr)
                row = row[:,0]
            row = row.reshape(row.size)
            observations.append(row)

        if event_table is not None:
            effects_normalized = np.zeros((days[-1] - days[0] + 1, len(event_to_int.values())))
            effects = np.zeros((len(days), len(event_to_int.values())))
            events = event_table[np.array(event_seq_id) == s_id,:]

            for event in events:
                event_idx = event[0]
                start_day = event[1]
                end_day = event[2]

                for day_idx, day in enumerate(range(days[0], days[-1] + 1)):
                    if start_day <= day and day <= end_day:
                        effects_normalized[day_idx,event_idx] = 1

                for day_idx, day in enumerate(days[:-1]):
                    start_idx = day - min(days)
                    end_idx = max(days[day_idx+1] - min(days), 1)
                    effects[day_idx] = effects_normalized[start_idx:end_idx,:].sum(axis=0) 
        else:
            effects = np.zeros((len(days), 1))
            effects_normalized = np.zeros((len(observations), 1))

        Y.append(np.array(observations))
        U.append(effects)
        T.append(days)

        U_normalized.append(effects_normalized)
        T_normalized.append(np.array([i for i in range(days[0], days[-1] + 1)]))

        assert effects_normalized.shape[0] == days.max() - days.min() + 1, str(effects_normalized) + "\n" + str(days)

    return Y, U, T, U_normalized, T_normalized


def load_observations(otu_filename, event_filename):
    """Read observations and transform to use with PoissonLDS.
    
    Parameters
    ----------
        otu_filename   : filepath of otu table
        event_filename : filepath of event table

    Returns
    -------
        obs_matrices   : a list of ObservationMatrix
    """

    otu_seq_id, otu_table = parse_otu_table(otu_filename, 
                                        sep=",",
                                        has_rownames=True)

    if event_filename != "":
        events, event_to_int, event_seq_id = parse_event_table(event_filename, sep=",")
    else:
        events = None
        event_seq_id = None
        event_to_int = None

    Y, U, T, U_expanded, T_expanded = format_observations(otu_table, otu_seq_id, events, event_seq_id, event_to_int)
    
    return Y, U, T, U_expanded, T_expanded



def print_error_and_die(error_msg):
    """Prints error_msg to stderr and halts program.
    """
    print(error_msg, file=sys.stderr)
    exit(1)