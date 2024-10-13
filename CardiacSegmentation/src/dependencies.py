"""
File: dependencies.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import qt
import slicer
import importlib
import platform


def qtMessage(title='', text='', critical_icon_flag=0):
    msg = qt.QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    if critical_icon_flag == 1:
        msg.setIcon(qt.QMessageBox.Critical)
    msg.exec_()


class Dependencies(object):
    """Docstring for Dependencies. """

    def __init__(self, dependencies=None):
        """ Initialization class to check and install the dependencies used for
        the Segmentation model

        Parameters
        ----------
        depnames : None or list of dependencies used for the module


        """
        super().__init__()
        self._platform = platform.system().lower()

        if dependencies is None:
            lib_names = ['skimage', 'nibabel', 'tensorflow']
            inst_name = ['scikit-image', 'nibabel', 'tensorflow']
            # TODO: tensorflow-macos seems not to be available for slicer
            #  if self._platform == 'darwin':
            #      lib_names[-1] = 'tensorflow-macos'
            #      inst_name[-1] = 'tensorflow-macos'
            #      lib_names.append('tensorflow-metal')
            #      inst_name.append('tensorflow-metal')
            dependencies = {
                k1: {
                    'installed': False,
                    'lib_name': k2
                }
                for k1, k2 in zip(inst_name, lib_names)
            }
        self._dep = dependencies
        self._installed = False

    def __iter__(self):
        return iter(list(self._dep))

    @property
    def libraries(self) -> dict:
        return self._dep

    @property
    def installed(self) -> bool:
        self.check()
        return self._installed

    @property
    def missing_libraries(self) -> list:
        self.check()
        return [k for k in self._dep if not self._dep[k]['installed']]

    @property
    def installed_libraries(self) -> list:
        self.check()
        return [k for k in self._dep if self._dep[k]['installed']]

    def check(self):
        if not self._installed:
            self._installed = True
            for k in self._dep:
                k2 = self._dep[k]['lib_name']
                if importlib.util.find_spec(k2) is not None:
                    self._dep[k]['installed'] = True
                else:
                    # Just in case we set to None
                    self._dep[k]['installed'] = False
                    self._installed = False

        return self._installed

    def install_libraries(self):
        qmsg = '%s is required by the CardIAc extension.'
        qmsg += ' Do you want to install it?'
        for k, val in self._dep.items():
            if not val['installed']:
                qm = qt.QMessageBox()
                ret = qm.question(None, '%s missing' % k, qmsg % k,
                                  qm.Yes | qm.No)
                if ret == qm.Yes:
                    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
                    try:
                        slicer.util.pip_install(k)
                        slicer.app.restoreOverrideCursor()
                        qtMessage('Succes', '%s is now installed!' % k)
                        self._dep[k]['installed'] = True
                    except Exception:
                        slicer.app.restoreOverrideCursor()
                        emsg = 'Error trying to install tensorflow. '
                        emsg += 'Try manual installation:\n'
                        emsg += "slicer.util.pip_install('%s')'" % val[
                            'lib_name']
                        qtMessage('Error', emsg, 1)
                        self._dep[k]['installed'] = False
                        self._installed = False
                if ret == qm.No:
                    emsg = "It won't be possible to use the CardIAC extension."
                    qtMessage('Warning', emsg, 1)
        # Update the status
        self.check()
