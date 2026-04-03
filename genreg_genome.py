# ================================================================
# GENREG Genome - Combines Controller + Proteins + Trust
# Payton Miller — 2025
# ================================================================

from genreg_controller import Controller
from hopf_controller import HopfController
from genreg_proteins import (
    SensorProtein, TrendProtein, TrustModifierProtein,
    ComparatorProtein, IntegratorProtein, GateProtein,
    run_protein_cascade
)


class Genome:
    """
    A complete GENREG genome with:
    - Regulatory layer (proteins)
    - Functional layer (controller NN)
    - Trust score (fitness)
    """

    def __init__(self, proteins=None, controller=None, signal_order=None,
                 controller_type="mlp", input_size=11, hidden_size=16, output_size=4):
        self.proteins = proteins if proteins else self._default_proteins()
        self.controller_type = controller_type
        if controller is not None:
            self.controller = controller
        elif controller_type == "hopf":
            self.controller = HopfController(input_size, hidden_size, output_size)
        else:
            self.controller = Controller(input_size, hidden_size, output_size)
        self.signal_order = signal_order  # Signal names in order (from Sensor)
        self.trust = 0.0
        self.lifetime_steps = 0
        self.food_eaten = 0

    def _default_proteins(self):
        """Create default protein network."""
        proteins = []

        # Sensor proteins for key signals
        proteins.append(SensorProtein("dist_to_food"))
        proteins.append(SensorProtein("energy"))
        proteins.append(SensorProtein("near_wall"))
        proteins.append(SensorProtein("steps_alive"))

        # Trend protein: track if getting closer to food
        trend_food = TrendProtein("trend_dist_food")
        trend_food.bind_inputs(["dist_to_food"])
        proteins.append(trend_food)

        # Trust modifier: reward getting closer (negative trend = closer = positive trust)
        trust_closer = TrustModifierProtein("trust_closer")
        trust_closer.bind_inputs(["trend_dist_food"])
        trust_closer.params["gain"] = -1.0  # Flip sign: closer = positive
        trust_closer.params["scale"] = 2.0
        proteins.append(trust_closer)

        # Trust modifier: reward survival
        trust_alive = TrustModifierProtein("trust_alive")
        trust_alive.bind_inputs(["steps_alive"])
        trust_alive.params["gain"] = 0.01
        trust_alive.params["scale"] = 1.0
        proteins.append(trust_alive)

        return proteins

    def step(self, signals):
        """
        Run one step: proteins process signals, update trust, controller selects action.

        Args:
            signals: dict from environment

        Returns:
            action: int (0-3)
            trust_delta: float
        """
        # Run protein cascade
        outputs, trust_delta = run_protein_cascade(self.proteins, signals)

        # Update trust
        self.trust += trust_delta
        self.lifetime_steps += 1

        # Controller selects action (requires signal_order)
        if not self.signal_order:
            raise ValueError("Genome.signal_order not set - must be provided from Sensor")
        action = self.controller.select_action(signals, self.signal_order)

        return action, trust_delta

    def reset(self):
        """Reset for new episode."""
        self.trust = 0.0
        self.lifetime_steps = 0
        self.food_eaten = 0

        # Reset protein states
        for p in self.proteins:
            p.reset_state()

    def mutate(self, rate=0.1, scale=0.3):
        """Mutate both controller and protein parameters."""
        self.controller.mutate(rate, scale)

        for p in self.proteins:
            for key in p.params:
                if isinstance(p.params[key], (int, float)):
                    p.mutate_param(key, scale)

    def clone(self):
        """Create a deep copy of this genome."""
        import copy
        new_proteins = []
        for p in self.proteins:
            new_p = copy.deepcopy(p)
            new_proteins.append(new_p)

        new_controller = self.controller.clone()
        g = Genome(proteins=new_proteins, controller=new_controller,
                   signal_order=self.signal_order,
                   controller_type=self.controller_type)
        g.trust = self.trust
        return g

    def to_dict(self):
        """Serialize genome to dictionary."""
        return {
            "trust": self.trust,
            "lifetime_steps": self.lifetime_steps,
            "food_eaten": self.food_eaten,
            "signal_order": self.signal_order,
            "controller_type": self.controller_type,
            "controller": self.controller.to_dict(),
            "proteins": [
                {
                    "type": p.__class__.__name__,
                    "name": p.name,
                    "params": p.params,
                    "inputs": p.inputs
                }
                for p in self.proteins
            ]
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize genome from dictionary."""
        controller_type = d.get("controller_type", "mlp")
        ctrl_data = d["controller"]
        if controller_type == "hopf" or ctrl_data.get("type") == "hopf":
            controller = HopfController.from_dict(ctrl_data)
            controller_type = "hopf"
        else:
            controller = Controller.from_dict(ctrl_data)
        signal_order = d.get("signal_order", None)

        protein_classes = {
            "SensorProtein": SensorProtein,
            "TrendProtein": TrendProtein,
            "TrustModifierProtein": TrustModifierProtein,
            "ComparatorProtein": ComparatorProtein,
            "IntegratorProtein": IntegratorProtein,
            "GateProtein": GateProtein
        }

        proteins = []
        for pd in d["proteins"]:
            ptype = pd["type"]
            if ptype == "SensorProtein":
                p = SensorProtein(pd["name"])
            else:
                p = protein_classes[ptype](pd["name"])
            p.params = pd["params"]
            p.inputs = pd["inputs"]
            proteins.append(p)

        g = cls(proteins=proteins, controller=controller, signal_order=signal_order,
                controller_type=controller_type)
        g.trust = d.get("trust", 0.0)
        g.lifetime_steps = d.get("lifetime_steps", 0)
        g.food_eaten = d.get("food_eaten", 0)
        return g
