# Initilize & Run

This page describes the key functions of reXplan. Information on data outputs are on the next site.

## Initialize Functions

The initialize model functions {py:func}`reXplan.simulation.Sim.initialize_model_rp` and
{py:func}`reXplan.simulation.Sim.initialize_model_sh` **create an outage schedule for each electrical equipment in the network as a database with timeseries**. It contains the operating information of all elements in the grid (“in service”, “out of service”, “awaiting repair”, “under repair”), which will be **used for the (Monte Carlo) simulation**.

---

In a **standard Monte Carlo simulation**, random samples of the hazard **intensity are selected from the entire range** of expected hazards. Since higher intensity events are less common, there will be a smaller representation of those events among the simulation range. To achieve better accuracy a large quantity of iterations must be selected. This is not always possible for bigger networks as it quickly becomes **computationally expensive**.

To counter this problem, a **stratification** method is used to divide the total range of possible intensities from the hazard into stratas with the objective being to minimize the total variance of the Monte Carlo samples within each strata. A separate Monte Carlo simulation can then be performed for each of the defined stratas with the number of samples as iterations. This method can achieve a **similar accuracy** compared to the Monte Carlo simulation without stratification with significantly fewer samples and thus **reducing the computational cost**.

Each Monte Carlo iteration will select a random intensity value within the strata intensity boundaries. The probability of failure of each equipment type is then determined using the fragility curves defined for each type of electrical equipment. A random probability value is then selected for each electrical equipment and is compared to the probability of failure to determine which elements will fail in that specific Monte Carlo iteration.

The time of failure is also selected randomly during the hazard period to avoid having all elements marked for failure, failing at the exact same time.

Using a queue system with priorities for different elements of the network, that can be defined by the user, a repair schedule is prepared. The repair schedule considers the number of crew available as well as the provided estimated repair times (time-to-repair= TTR) for each of the electrical equipment.

The equipment failure schedule in combination with the repair schedule form the *outage schedule*. For each Monte Carlo iteration one outage schedule is generated. The sum of all schedules is stored in the [`montecarlo_database.csv`](./out_monte_carlo.md) file.

## Run Function


The run function {py:func}`reXplan.simulation.Sim.run` executes the Monte Carlo simulation based on the [`montecarlo_database.csv`](./out_monte_carlo.md). Calculates the OPF (Optimal Power Flow) and stores data in [`engine_database.csv`](./out_engine.md)

---

The OPF is used to calculate the load curtailments necessary in each operating scenario, to avoid loading violations. The use of Julia programming language for OPF greatly reduced the computation time, compared to Python. The complete results of the simulations are stored in the tool as a single large multi-indexed data frame that can be locally saved and loaded in any data science framework, to flexibly extract and analyse the simulation results.

[#TODO How many Iterations per MC Database?]