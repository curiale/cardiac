from slicer.ScriptedLoadableModule import ScriptedLoadableModule
from Widget.CardiacStrainWidget import CardiacStrainWidget  # noqa

class CardiacStrain(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CardIAc Strain"
        self.parent.categories = ["CardIAc"]
        self.parent.dependencies = []
        self.parent.contributors = ["Agustin Bernardo, Ariel Curiale"]
        self.parent.helpText = """
        This module measures Displacement Field and analyzes Strain
        from a CineMRI sequence and it's Myocardial segmentation,
        to quantify cardiac function.
        """
        self.parent.acknowledgementText = """
        This module was created by M.Sc. A. Bernardo and Ph.D. A. H. Curiale,
        with help from M.Sc. L. Dellazzoppa, at the Medical Physics Department,
        from the Atomic Center of Bariloche, Argentina.
        """
