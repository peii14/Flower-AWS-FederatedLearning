import flwr as fl
from typing import List,Tuple
import numpy as np
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
strategy = AggregateCustomMetricStrategy(
    fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
    min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start
)

fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=strategy)

