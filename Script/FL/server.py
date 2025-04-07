import flwr as fl
import argparse
from utils import is_interactive
from custom_server import ScaffoldServer
from SCAFFOLD_strategy import ScaffoldStrategy



parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--server-address", type=str, default="localhost:12389", help="Address of the server")
parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")

if is_interactive():
  args, _ = parser.parse_known_args()
else:
  args = parser.parse_args()

NUM_ROUNDS = args.num_rounds
print(f"NUM_ROUNDS: {NUM_ROUNDS}")
min_fit_clients = 12

status_file = "status_server.txt"
open(status_file, "a").close()

print(f"Server starting with {NUM_ROUNDS} rounds and min_fit_clients {min_fit_clients}")

# FedAvg - FedProx strategy
strategy= fl.server.strategy.FedProx(
  min_fit_clients=min_fit_clients,
  min_evaluate_clients=min_fit_clients,
  min_available_clients=min_fit_clients,
  proximal_mu=0.9 # FedProx with proximal_mu == 0 is equivalent to FedAvg
)

# SCAFFOLD 
# server=ScaffoldServer(
#   strategy=ScaffoldStrategy(
#     min_fit_clients=min_fit_clients,
#     min_evaluate_clients=min_fit_clients,
#     min_available_clients=min_fit_clients,
#   ),
# )


# Start Flower server
fl.server.start_server(grpc_max_message_length = 1_074_422_052,
  server_address=args.server_address,
  config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
  strategy = strategy,
)
