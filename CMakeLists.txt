cmake_minimum_required(VERSION 3.5)

project(CardIAc)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CardIAc")
set(EXTENSION_CATEGORY "CardIAc")
set(EXTENSION_CONTRIBUTORS "Agustín Bernardo, Lucca Dellazoppa, Ariel Curiale (Instituto Balseiro)")
set(EXTENSION_DESCRIPTION "Myocardial segmentation and cardiac quantification.")
set(EXTENSION_ICONURL "CardIAc.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/CardIAc/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CardIAcSegmentation)
add_subdirectory(CardIAcStrain)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
