# Naming Conventions for This Project
## [Reference](http://visualgit.readthedocs.io/en/latest/pages/naming_convention.html)

### 1. General
Avoid using names that are too general or too wordy. Strike a good balance between the two.<br />
Bad: data_structure, my_list, info_map, dictionary_for_the_purpose_of_storing_data_representing_word_definitions<br />
Good: user_profile, menu_options, word_definitions<br />
Don’t be a jackass and name things “O”, “l”, or “I”<br />
When using CamelCase names, capitalize all letters of an abbreviation (e.g. HTTPServer)<br />

### 2. Packages
Package names should be all lower case<br />
When multiple words are needed, an underscore should separate them<br />
It is usually preferable to stick to 1 word names<br />

### 3. Modules
Module names should be all lower case<br />
When multiple words are needed, an underscore should separate them<br />
It is usually preferable to stick to 1 word names<br />

### 4. Classes
Class names should follow the UpperCaseCamelCase convention<br />
Python’s built-in classes, however are typically lowercase words<br />
Exception classes should end in “Error”<br />

### 5. Global (module-level) Variables
Global variables should be all lowercase<br />
Words in a global variable name should be separated by an underscore<br />

### 6. Instance Variables
Instance variable names should be all lower case<br />
Words in an instance variable name should be separated by an underscore<br />
Non-public instance variables should begin with a single underscore<br />
If an instance name needs to be mangled, two underscores may begin its name<br />

### 7. Methods
Method names should be all lower case<br />
Words in an method name should be separated by an underscore<br />
Non-public method should begin with a single underscore<br />
If a method name needs to be mangled, two underscores may begin its name<br />

### 8. Method Arguments
Instance methods should have their first argument named ‘self’.<br />
Class methods should have their first argument named ‘cls’<br />

### 9. Functions
Function names should be all lower case<br />
Words in a function name should be separated by an underscore<br />

### 10. Constants
Constant names must be fully capitalized<br />
Words in a constant name should be separated by an underscore<br />
