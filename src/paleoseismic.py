import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal
import copy
import seaborn as sns

# TODO:
# Fit statistical model to the interevents times
#   - Compare average fit to joint fit
# Sample records form the fit distributions (prediction that the effects should be more severe)
# Mask out intervals without data
# Ideas for how to deal with potential missing events deeper in the past


class PaleoseismicCatalog:
    def __init__(
        self,
        file: str,
        detect_order: bool = True,
        start_time: float | None = None,
        end_time: float | None = 2025,
        minimum_interevent_time: float = 10,  # years
        historic_events: List[float] | None = None,
        gaps: List[List[float]] | None = None,
        notes: str | None = None,
        references: List[str] | None = None,
    ):
        self.file = file
        self._raw_record = self.load_data()
        self.record = copy.deepcopy(self._raw_record)

        self.notes = notes
        self.references = references

        if self.detect_time_order() != "increasing" and detect_order:
            self.reverse_events()

        self.minimum_interevent_time = minimum_interevent_time
        
        self.historic_events = historic_events if historic_events else []
        self.try_event_attribution()
        
        self.gaps = gaps if gaps else []

        self.__start_time = start_time
        self.__end_time = end_time

        # NOTE: consider padding the record or enforcing a start and end time
        # for short records this might result is biasing the the sampling

        self.check()

        self.__update__()

    def try_event_attribution(self, year_buffer: float = 1.0, method: Literal["countback", "max likelihood"] = "countback"):
        
        if len(self.historic_events) > 1:
            assert np.min(np.diff(np.sort(self.historic_events))) > self.minimum_interevent_time, f"The edge case of historic events < {self.minimum_interevent_time} years apart has not been implemented yet"
        
        # note that the record could be altered if the code crashes half way through
        old_record = copy.deepcopy(self.record)  # in case things go wrong

        reversed_historic_events = np.sort(self.historic_events)[::-1]
        for historical_event_year in reversed_historic_events:
            p = []
            for event in self.get_events():
                age_bool = (event["age"] >= historical_event_year - year_buffer) & (
                    event["age"] <= historical_event_year + year_buffer
                )

                p.append(sum(event["PDF"][age_bool]) if age_bool.any() else 0)
            
            p = np.array(p)

            if sum(p) > 0:
                
                if method == "max likelihood":
                    selected_event_index = np.argmax(p)
                elif method == "countback":
                    selected_event_index = np.argmax(np.cumsum(p>0)) # i.e. the last non-zero probability, idk life is hard
                    # note that this works for multiple events because self.record is changing
                    
                event_code = self.get_event_codes()[selected_event_index]

                new_event = pd.DataFrame(
                    {
                        "event": [event_code, event_code, event_code],
                        "age": [
                            historical_event_year - 1,
                            historical_event_year,
                            historical_event_year + 1,
                        ],
                        "PDF": [0, 1, 0],
                        "CDF": [0, 1, 1],
                    }
                )
                
                replace_index = selected_event_index
                
                events = self.get_events()

                parts = []
                if replace_index > 0:
                    parts.extend(events[:replace_index])

                parts.append(new_event) 

                if replace_index < len(events)-1:
                    parts.extend(events[replace_index+1:])

                self.record = pd.concat(parts, ignore_index=True)

            else:
                print(
                    f"No event found for {historical_event_year}, inserting historic event assuming it was not recorded in the trench"
                )
                # insert a new event with age = historical_event_year and PDF = 1
                new_event_code = f"Historical Event {historical_event_year}"
                new_event = pd.DataFrame(
                    {
                        "event": 3 * [new_event_code],
                        "age": [
                            historical_event_year - 1,
                            historical_event_year,
                            historical_event_year + 1,
                        ],
                        "PDF": [0, 1, 0],
                        "CDF": [0, 1, 1],
                    }
                )

                # insert event into the record
                average_ages = [e["age"].mean() for e in self.get_events()]
                insert_index = np.argmax(historical_event_year < average_ages)
                
                events = self.get_events()

                parts = []
                if insert_index > 0:
                    parts.append(events[:insert_index])

                parts.append(new_event) 

                if insert_index < len(events) - 1:
                    parts.append(events[insert_index:])

                self.record = pd.concat(*parts, ignore_index=True)

                # edge case:
        # check order attribution could, in principle break the event order:
        # E2    #
        #    #######
        # #############!#
                    
        # E1     #     |
        #        #     
        #       ###### x

        # H            |
        #              |
        #              |
        #              |

        for i in range(len(self.get_events())-1):
            try:
                assert not (
                    self.get_events()[i]["age"].min()
                    > self.get_events()[i + 1]["age"].max()
                )
            except AssertionError:
                print("Cannot reconcile historical events, returning to orignal record without historical events, consider fixing manually")
                self.record = old_record

    def __update__(self):
        self.record = self.record[self.record.age > self.__start_time]
        self.record = self.record[self.record.age < self.__end_time]
        self.gaps = [
            [max(self.__start_time, t1), t2]
            for t1, t2 in self.gaps
            if t2 > self.__start_time
        ]
        self.gaps = [
            [t1, min(t2, self.__end_time)]
            for t1, t2 in self.gaps
            if t1 < self.__end_time
        ]
        self.number_of_events = np.sum(self.record["PDF"])
        self.gap_time = sum([t2 - t1 for t1, t2 in self.gaps])
        self.average_interval = (
            self.end_time - self.start_time - self.gap_time
        ) / self.number_of_events

    @property
    def start_time(self):
        if self.__start_time is None:
            self.__start_time = np.min(self.record["age"])
        return self.__start_time

    @start_time.setter
    def start_time(self, value):
        self.__start_time = value
        self.__update__()

    @property
    def end_time(self):
        if self.__end_time is None:
            self.__end_time = np.max(self.record["age"])

        return self.__end_time

    @end_time.setter
    def end_time(self, value):
        self.__end_time = value
        self.__update__()

    def load_data(self):
        return pd.read_csv(self.file, delimiter="\t")

    def detect_time_order(self):
        if np.all(np.diff(self.get_expected_times()) >= 0):
            event_order = "increasing"
        elif np.all(np.diff(self.get_expected_times()) <= 0):
            event_order = "decreasing"
        else:
            raise NotImplementedError("events are neither all increasing or decreasing")

        return event_order

    def check(self):
        for event in self.get_events():
            assert "age" in event.columns, "event must contain age column"
            assert "PDF" in event.columns, "event must contain PDF column"
        assert self.start_time < self.end_time

        if self.gaps:
            for t1, t2 in self.gaps:
                assert (
                    t1 < t2
                ), "gaps must be have start time less than end time and be enclosed in a list"

    def get_expected_times(self) -> np.ndarray:
        return np.array(
            [(event["age"] * event["PDF"]).sum() for event in self.get_events()]
        )

    def get_events(self) -> List[pd.DataFrame]:
        return [e[1] for e in self.record.groupby("event", sort=False)]

    def get_event_codes(self) -> List[str]:
        return [e[0] for e in self.record.groupby("event", sort=False)]

    def reverse_events(self):
        self.record = pd.concat(self.get_events()[::-1])

    def _add_gaps(self, ax):
        if self.gaps:
            [ax.axvspan(t1, t2, color="red", alpha=0.2) for t1, t2 in self.gaps]

    def plot_pdfs(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 2))

        for event, event_code in zip(
            self.get_events(), self.get_event_codes()
        ):  # important: pandas (default) sorting causes the ages to be out of order
            mid_points = 0.5 * (event["age"].values[1:] + event["age"].values[:-1])
            ax.plot(
                mid_points,
                event["PDF"].iloc[:-1] / np.diff(event["age"]),
                label=event_code,
                lw=0.5,
            )

        self._add_gaps(ax)

        ax.set(
            ylim=(0, 0.03),
            xlabel="Calendar year",
            ylabel="Probability density",
            xlim=(self.start_time, self.end_time),
        )

        return ax

    def visualize_order(self, ax=None):

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        t_max = []
        t_mean = []
        for event in self.get_events():
            age = event["age"].values
            pdf = event["PDF"]
            valid_age = age[pdf > 0]
            t_max.append(np.max(valid_age))
            t_mean.append(np.sum(pdf * age))

        ax.plot(np.arange(len(t_max)), t_max, label="max time")
        ax.plot(np.arange(len(t_max)), t_mean, label="mean time")
        ax.set(
            xlabel="index",
            ylabel="age",
            xticks=np.arange(len(t_max)),
        )
        ax.set_xticklabels(self.get_event_codes(), rotation=90)

        ax.legend()

        return ax

    def visualize_sampling(self, number_of_samples=100, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 2))

        sample_list = []
        for _ in range(number_of_samples):
            sample = self.sample()
            sample_list.append(sample)
            [
                (
                    ax.axvline(t, c=f"C{i%10}", alpha=0.1, lw=0.5)
                    if not np.isnan(t)
                    else None
                )
                for i, t in enumerate(sample)
            ]

        for i in range(np.ceil(self.number_of_events).astype(int)):
            sns.kdeplot(
                [s[i] for s in sample_list],
                label=f"Event {i}",
                ls="--",
                lw=1,
                bw_adjust=0.5,
                ax=ax,
            )

        self._add_gaps(ax)

    def visualize_waiting_time(self, number_of_samples=100, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        waiting_time = []
        lapse_time = []
        for _ in range(number_of_samples):
            sample_event_times = self.sample()

            i = 0
            max_attempts = 1000
            while True:
                t0 = (
                    np.random.random() * (self.end_time - self.start_time)
                    + self.start_time
                )
                candidate_waiting_time = self.get_waiting_time(t0, sample_event_times)
                candidate_lapse_time = self.get_lapse_time(t0, sample_event_times)

                if not np.isnan(candidate_waiting_time) and not np.isnan(
                    candidate_lapse_time
                ):
                    break
                elif i > max_attempts:
                    raise ValueError(
                        f"Could not find a sample time in bounds {self.start_time}, {self.end_time} in {max_attempts} attempts. "
                        "Gaps may be too big."
                    )

            waiting_time.append(candidate_waiting_time)
            lapse_time.append(candidate_lapse_time)

        waiting_time = np.array(waiting_time)
        lapse_time = np.array(lapse_time)

        ax.hist(waiting_time, bins=100)
        ax.axvline(np.nanmean(waiting_time), c="r", ls="--", label="Average wait time")
        ax.axvline(
            np.nanmean(waiting_time + lapse_time),
            c="r",
            label="Average interval (random observer)",
        )
        ax.axvline(
            self.average_interval / 2, c="k", ls="--", label="1/2 Average interval"
        )
        ax.axvline(self.average_interval, c="k", label="Average interval")

        ax.legend(bbox_to_anchor=(1,1))

        return ax, waiting_time, lapse_time

    def visualize_paradox(self, number_of_samples=100, ax=None):
        if ax is None:
            _, AX = plt.subplots(2, 1, sharex=True)

        waiting_time = []
        lapse_time = []
        random_interval = []
        for _ in range(number_of_samples):

            sample_event_times = self.sample()

            # get the elapsed and waiting time for random start times
            # rejects sample observer times if the point in time is immidiately prior or after a gap
            i = 0
            max_attempts = 1000
            while True:
                t0 = (
                    np.random.random() * (self.end_time - self.start_time)
                    + self.start_time
                )
                candidate_waiting_time = self.get_waiting_time(t0, sample_event_times)
                candidate_lapse_time = self.get_lapse_time(t0, sample_event_times)

                if not (
                    np.isnan(candidate_waiting_time) and np.isnan(candidate_lapse_time)
                ):
                    break
                elif i > max_attempts:
                    raise ValueError(
                        f"Could not find a sample time in bounds {self.start_time}, {self.end_time} in {max_attempts} attempts. "
                        "Gaps may be too big."
                    )

            waiting_time.append(candidate_waiting_time)
            lapse_time.append(candidate_lapse_time)

            # get a random interevent time from the sampled timeseries
            random_interval.append(
                np.random.choice(self.get_interevent_times(sample_event_times))
            )

        # note waiting time or lapse time may contain nan
        waiting_time = np.array(waiting_time)
        lapse_time = np.array(lapse_time)
        random_interval = np.array(random_interval)

        random_observer = waiting_time + lapse_time
        event_based_observer = random_interval

        ax = AX[0]
        hist_kwargs = dict(alpha=0.5, density=True)
        random_observer_density, bins, _ = ax.hist(
            random_observer, bins=50, label="Random observer (us)", **hist_kwargs
        )
        event_based_observer_density, _, _ = ax.hist(
            event_based_observer, bins=bins, label="Inteval likelihood", **hist_kwargs
        )
        ax.legend()

        # make sure that the counts are the same:
        bin_width = np.diff(bins)
        random_observer_probability_mass = random_observer_density * bin_width
        event_based_probability_mass = event_based_observer_density * bin_width

        ax.set(yticks=[])

        random_observer_survival = 1 - np.cumsum(random_observer_probability_mass)
        event_based_survival = 1 - np.cumsum(event_based_probability_mass)

        axb = AX[1]
        axb.plot(
            0.5 * (bins[1:] + bins[:-1]),
            random_observer_survival / event_based_survival,
            c="r",
        )

        axb.set(
            xlabel="Interval length (years)",
            ylabel="bias \n $P_r(T>T_i)/P_e(T>T_i)$",
        )

        axb.axhline(1, c="k", ls="--")

        plt.tight_layout()

        return AX

    @staticmethod
    def sample_event(event, n=1):
        """Returns an age sample from the age distribution.
        !!!Note returns np.nan with probability 1 - sum(event["PDF"])!!!
        """

        assert "age" in event.columns, "event must contain age column"
        assert "PDF" in event.columns, "event must contain PDF column"

        event_probability = np.sum(event["PDF"])

        age_samples = np.random.choice(
            event["age"], size=n, p=event["PDF"] / event_probability
        )
        reject = np.random.random(n) > event_probability
        age_samples[reject] = np.nan

        return age_samples

    def sample(self):
        while (
            True
        ):  # brute force - will take a long time for records events with heavily overlapping PDFs
            candidate_sample = np.concatenate(
                [self.sample_event(event) for event in self.get_events()]
            )
            if np.all(
                np.diff(candidate_sample[~np.isnan(candidate_sample)])
                > self.minimum_interevent_time
            ):
                valid_sample = candidate_sample
                break
        return valid_sample

    def get_waiting_time(self, t0: float, t: np.ndarray, use_gaps: bool = True):
        dt = t - t0
        positive_dt = dt[dt > 0]

        waiting_time = np.min(positive_dt) if np.any(positive_dt) else np.nan

        # annoyingly complex code to avoid gaps or intervals up to gaps
        # Concern: does this condition the waiting time on being far from the gap?
        if use_gaps and not np.isnan(waiting_time):
            for g1, g2 in self.gaps:
                if g1 < t0 < g2:  # inside the gap
                    waiting_time = np.nan
                    break
                elif t0 < g1 < t0 + waiting_time:
                    waiting_time = np.nan
                    break

        return waiting_time

    def get_lapse_time(self, t0: float, t: np.ndarray, use_gaps: bool = True):
        dt = t - t0
        negative_dt = np.abs(dt[dt < 0])

        lapse_time = np.min(negative_dt) if np.any(negative_dt) else np.nan

        # annoyingly complex code to avoid gaps or intervals since gaps
        if use_gaps and not np.isnan(lapse_time):
            for g1, g2 in self.gaps:
                if g1 < t0 < g2:
                    lapse_time = np.nan
                    break
                if t0 - lapse_time < g2 < t0:
                    lapse_time = np.nan
                    break

        return lapse_time

    @staticmethod
    def get_overlap(a, b):
        """
        Calculates the overlap between interval a and b. Returns 0 if there is no overlap
        """
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def get_interevent_times(self, t, exclude_gaps: bool = True):
        """
        Calculates the interevent times for the series `t`. If `exclude_gaps` is true, each interval is
        checked to ensure that there is no overlap with the gaps.
        """

        dt = np.diff(t)

        if exclude_gaps is True:
            contains_gap = []
            for t1, t2 in zip(t[:-1], t[1:]):
                contains_gap.append(
                    1
                    if np.any(
                        [self.get_overlap([t1, t2], i_gap) for i_gap in self.gaps]
                    )
                    else 0
                )
            dt = dt[np.logical_not(contains_gap)]

        return dt

    @staticmethod
    def COV(dt):
        return np.std(dt) / np.mean(dt)

    def wait_time_bias(self, dt):
        return np.mean(dt) / 2 * (self.COV(dt) ** 2 + 1)

    def observer_aware_interevent_times(
        self,
        number_of_samples: int = 100,
        observer: Literal["event", "random"] = "event",
    ):
        if observer == "event":
            batches = []
            i = 0
            while i < number_of_samples:
                batch = self.get_interevent_times(self.sample())
                batches.append(batch)
                i += len(batch)

            tau = np.concatenate(batches)[:number_of_samples]

        if observer == "random":
            i = 0
            tau = []
            while i < number_of_samples:
                t = self.sample()
                t0 = np.random.uniform(self.start_time, self.end_time)
                t1 = self.get_lapse_time(t0, t)
                t2 = self.get_waiting_time(t0, t)
                if not np.isnan(
                    t1 + t2
                ):  # note that for short records censoring around gaps would bias the wait time
                    tau.append(t1 + t2)
                    i += 1
            tau = np.array(tau)

        assert len(tau) == number_of_samples

        return tau

    def ecdf(
        self,
        Tq,
        number_of_samples: int = 100,
        observer: Literal["event", "random"] = "event",
    ):

        Tq = np.atleast_1d(Tq)

        # efficient search
        tau = np.sort(self.observer_aware_interevent_times(number_of_samples, observer))
        counts = np.searchsorted(tau, Tq, side="right")
        ecdf_vals = counts / number_of_samples

        return ecdf_vals if Tq.size > 1 else ecdf_vals.item()
