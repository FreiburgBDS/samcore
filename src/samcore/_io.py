# samcore._io - file I/O helpers (backed by the C++ io submodule).

from samcore import _samcore


class io: 
    read_h5sam = staticmethod(_samcore.io.read_h5sam)
    write_h5sam = staticmethod(_samcore.io.write_h5sam)
    read_h5samd = staticmethod(_samcore.io.read_h5samd)
    convert_h5sam_to_h5samd = staticmethod(_samcore.io.convert_h5sam_to_h5samd)
