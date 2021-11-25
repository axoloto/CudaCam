from conans import ConanFile


class CppStarterProject(ConanFile):
    # Note: options are copied from CMake boolean options.
    # When turned off, CMake sometimes passes them as empty strings.
    options = {
        "use_imgui": ["ON", "OFF", ""]
    }
    name = "CppStarterProject"
    version = "0.1"
    requires = (
        "catch2/2.13.7",
        "docopt.cpp/0.6.2",
        "fmt/8.0.1",
        "spdlog/[>=1.9.2]",
        "opencv/[>=4.5.3]"
    )
    generators = "cmake", "gcc", "txt", "cmake_find_package"

    def requirements(self):
        if self.options.use_imgui == "ON":
            self.requires("sdl/[>=2.0.12]")
            self.requires("imgui/[>=1.79]")

