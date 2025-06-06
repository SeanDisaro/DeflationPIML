from logging_config import setup_logging
#from tests.Reduced2DimLDG_ApproxDerivativeWithNN.run import run
#from tests.Reduced2DimLDG_AutoGradDerivative.run import run
from tests.Reduced2DimLDG_FixedBranchFeatures.run import run

logger = setup_logging()

#logger = logging.getLogger(__name__)


if __name__ == "__main__":
    run()