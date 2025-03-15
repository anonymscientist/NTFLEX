from TFLEX.assistance.toolbox.exp.OutputSchema import OutputSchema
from TFLEX.assistance.toolbox.utils.ModelParamStore import ModelParamStoreSchema
from TFLEX.assistance.toolbox.utils.VisualizeStore import VisualizeStoreSchema


class Self:
    pass


def init_by_output(self: object, output: OutputSchema):
    self.debug = output.logger.debug
    self.log = output.logger.info
    self.warn = output.logger.warn
    self.error = output.logger.error
    self.critical = output.logger.critical
    self.success = output.logger.success
    self.fail = output.logger.failed
    self.vis = VisualizeStoreSchema(str(output.pathSchema.dir_path_visualize))
    self.store = ModelParamStoreSchema(output.pathSchema)
