# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bury/project/AI_gomoku

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bury/project/AI_gomoku/build

# Include any dependencies generated for this target.
include CMakeFiles/ai_cmake.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ai_cmake.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ai_cmake.dir/flags.make

CMakeFiles/ai_cmake.dir/main.cpp.o: CMakeFiles/ai_cmake.dir/flags.make
CMakeFiles/ai_cmake.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ai_cmake.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ai_cmake.dir/main.cpp.o -c /home/bury/project/AI_gomoku/main.cpp

CMakeFiles/ai_cmake.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ai_cmake.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bury/project/AI_gomoku/main.cpp > CMakeFiles/ai_cmake.dir/main.cpp.i

CMakeFiles/ai_cmake.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ai_cmake.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bury/project/AI_gomoku/main.cpp -o CMakeFiles/ai_cmake.dir/main.cpp.s

CMakeFiles/ai_cmake.dir/src/Man.cpp.o: CMakeFiles/ai_cmake.dir/flags.make
CMakeFiles/ai_cmake.dir/src/Man.cpp.o: ../src/Man.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ai_cmake.dir/src/Man.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ai_cmake.dir/src/Man.cpp.o -c /home/bury/project/AI_gomoku/src/Man.cpp

CMakeFiles/ai_cmake.dir/src/Man.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ai_cmake.dir/src/Man.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bury/project/AI_gomoku/src/Man.cpp > CMakeFiles/ai_cmake.dir/src/Man.cpp.i

CMakeFiles/ai_cmake.dir/src/Man.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ai_cmake.dir/src/Man.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bury/project/AI_gomoku/src/Man.cpp -o CMakeFiles/ai_cmake.dir/src/Man.cpp.s

CMakeFiles/ai_cmake.dir/src/AI.cpp.o: CMakeFiles/ai_cmake.dir/flags.make
CMakeFiles/ai_cmake.dir/src/AI.cpp.o: ../src/AI.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ai_cmake.dir/src/AI.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ai_cmake.dir/src/AI.cpp.o -c /home/bury/project/AI_gomoku/src/AI.cpp

CMakeFiles/ai_cmake.dir/src/AI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ai_cmake.dir/src/AI.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bury/project/AI_gomoku/src/AI.cpp > CMakeFiles/ai_cmake.dir/src/AI.cpp.i

CMakeFiles/ai_cmake.dir/src/AI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ai_cmake.dir/src/AI.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bury/project/AI_gomoku/src/AI.cpp -o CMakeFiles/ai_cmake.dir/src/AI.cpp.s

CMakeFiles/ai_cmake.dir/src/Chess.cpp.o: CMakeFiles/ai_cmake.dir/flags.make
CMakeFiles/ai_cmake.dir/src/Chess.cpp.o: ../src/Chess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ai_cmake.dir/src/Chess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ai_cmake.dir/src/Chess.cpp.o -c /home/bury/project/AI_gomoku/src/Chess.cpp

CMakeFiles/ai_cmake.dir/src/Chess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ai_cmake.dir/src/Chess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bury/project/AI_gomoku/src/Chess.cpp > CMakeFiles/ai_cmake.dir/src/Chess.cpp.i

CMakeFiles/ai_cmake.dir/src/Chess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ai_cmake.dir/src/Chess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bury/project/AI_gomoku/src/Chess.cpp -o CMakeFiles/ai_cmake.dir/src/Chess.cpp.s

CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o: CMakeFiles/ai_cmake.dir/flags.make
CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o: ../src/ChessGame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o -c /home/bury/project/AI_gomoku/src/ChessGame.cpp

CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bury/project/AI_gomoku/src/ChessGame.cpp > CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.i

CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bury/project/AI_gomoku/src/ChessGame.cpp -o CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.s

# Object files for target ai_cmake
ai_cmake_OBJECTS = \
"CMakeFiles/ai_cmake.dir/main.cpp.o" \
"CMakeFiles/ai_cmake.dir/src/Man.cpp.o" \
"CMakeFiles/ai_cmake.dir/src/AI.cpp.o" \
"CMakeFiles/ai_cmake.dir/src/Chess.cpp.o" \
"CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o"

# External object files for target ai_cmake
ai_cmake_EXTERNAL_OBJECTS =

ai_cmake: CMakeFiles/ai_cmake.dir/main.cpp.o
ai_cmake: CMakeFiles/ai_cmake.dir/src/Man.cpp.o
ai_cmake: CMakeFiles/ai_cmake.dir/src/AI.cpp.o
ai_cmake: CMakeFiles/ai_cmake.dir/src/Chess.cpp.o
ai_cmake: CMakeFiles/ai_cmake.dir/src/ChessGame.cpp.o
ai_cmake: CMakeFiles/ai_cmake.dir/build.make
ai_cmake: CMakeFiles/ai_cmake.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bury/project/AI_gomoku/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ai_cmake"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ai_cmake.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ai_cmake.dir/build: ai_cmake

.PHONY : CMakeFiles/ai_cmake.dir/build

CMakeFiles/ai_cmake.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ai_cmake.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ai_cmake.dir/clean

CMakeFiles/ai_cmake.dir/depend:
	cd /home/bury/project/AI_gomoku/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bury/project/AI_gomoku /home/bury/project/AI_gomoku /home/bury/project/AI_gomoku/build /home/bury/project/AI_gomoku/build /home/bury/project/AI_gomoku/build/CMakeFiles/ai_cmake.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ai_cmake.dir/depend
