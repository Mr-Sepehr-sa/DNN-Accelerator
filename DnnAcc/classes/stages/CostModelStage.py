from typing import Generator, Callable, List, Tuple, Any

from DnnAcc.classes.stages.Stage import Stage
from DnnAcc.classes.cost_model.cost_model import CostModelEvaluation
from DnnAcc.classes.hardware.architecture.accelerator import Accelerator
from DnnAcc.classes.mapping.spatial.spatial_mapping import SpatialMapping
from DnnAcc.classes.mapping.temporal.temporal_mapping import TemporalMapping
from DnnAcc.classes.workload.layer_node import LayerNode

import logging
logger = logging.getLogger(__name__)


class CostModelStage(Stage):
    """
    Pipeline stage that calls a cost model to evaluate a mapping on a HW config.
    """
    def __init__(self, list_of_callables:List[Callable], *, accelerator, layer, spatial_mapping, temporal_mapping, **kwargs):
        """
        Initializes the cost model stage given main inputs
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping, self.temporal_mapping =\
            accelerator, layer, spatial_mapping, temporal_mapping

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the cost model stage by calling the internal DnnAcc cost model with the correct inputs.
        """
        self.cme = CostModelEvaluation(accelerator=self.accelerator,
                                       layer=self.layer,
                                       spatial_mapping=self.spatial_mapping,
                                       temporal_mapping=self.temporal_mapping)
        yield (self.cme, None)

    def is_leaf(self) -> bool:
        return True