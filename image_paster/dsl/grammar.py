"""Lark EBNF grammar for C++ style Semantic Scene DSL."""

CPP_SCENE_DSL_GRAMMAR = r'''
start: scene

scene: "scene" [NAME] "{" scene_body* "}"

scene_body: camera_block
          | environment_block
          | objects_block
          | shapes_block
          | struct_def
          | edits_block
          | relations_block
          | constraints_block
          | operations_block

camera_block: "camera" "{" assignment* "}"

environment_block: "environment" "{" env_item* "}"
env_item: assignment
        | lighting_block
        | search_call_stmt

lighting_block: "lighting" "{" assignment* "}"

objects_block: "objects" "{" (object_def | struct_def | shape_def)* "}"
shapes_block: "shapes" "{" shape_def* "}"

object_def: "object" NAME "{" object_item* "}" -> full_object_def
          | "object" NAME "=" search_call ["{" object_item* "}"] [";"] -> object_search_def
          | "object" NAME "=" copy_call ["{" object_item* "}"] [";"] -> object_copy_def
          | "object" NAME "=" linspace_call ["{" object_item* "}"] [";"] -> object_linspace_def
          | "object" NAME "=" summon_call ["{" object_item* "}"] [";"] -> object_summon_def
          | "object" NAME "=" struct_call ["{" object_item* "}"] [";"] -> object_struct_def
          | "object" NAME "=" shape_call ["{" object_item* "}"] [";"] -> object_shape_def
          | NAME "=" search_call ["{" object_item* "}"] [";"] -> shorthand_search_def
          | NAME "=" copy_call ["{" object_item* "}"] [";"] -> shorthand_copy_def
          | NAME "=" linspace_call ["{" object_item* "}"] [";"] -> shorthand_linspace_def
          | NAME "=" summon_call ["{" object_item* "}"] [";"] -> shorthand_summon_def
          | NAME "=" struct_call ["{" object_item* "}"] [";"] -> shorthand_struct_def
          | NAME "=" shape_call ["{" object_item* "}"] [";"] -> shorthand_shape_def
          | linspace_call [";"] -> standalone_linspace
          | summon_call [";"] -> standalone_summon
          | shape_call [";"] -> standalone_shape

shape_def: shape_type NAME ["(" [shape_args] ")"] ["{" shape_item* "}"] [";"] -> direct_shape_def
         | "object" NAME "=" shape_call ["{" object_item* "}"] [";"] -> object_shape_def
         | NAME "=" shape_call ["{" object_item* "}"] [";"] -> shorthand_shape_def
         | shape_call [";"] -> standalone_shape

!shape_type: "circle" | "rectangle" | "triangle" | "line" | "curve" | "text" | "ellipse" | "polygon"

shape_call: shape_type "(" [shape_args] ")"
shape_args: value ("," value)*
shape_item: object_item

struct_def: "struct" NAME "{" struct_item* "}" [";"]
struct_item: assignment
           | object_item

linspace_call: "linspace" "(" object_target "," value ")"
summon_call: "summon" "(" object_target "," value ")"
object_target: shape_call | struct_call | search_call | copy_call | NAME

struct_call: "struct" "(" struct_arg "," struct_arg ("," struct_arg)* ")"
struct_arg: shape_call | struct_call | search_call | copy_call | ESCAPED_STRING | NAME

object_item: source_block
           | appearance_block
           | transform_block
           | lighting_block
           | search_call_stmt
           | chained_call
           | method_stmt
           | assignment

source_block: "source" "{" source_item* "}"
source_item: assignment
           | search_call_stmt

search_call_stmt: search_call ";"

search_call: "search" "(" ESCAPED_STRING ("," ESCAPED_STRING)* ")"

copy_call: "copy" "(" NAME ")"

appearance_block: "appearance" "{" assignment* "}"
transform_block: "transformation" "{" assignment* "}"

assignment: NAME "=" value ";"

edits_block: "edits" "{" edit_item* "}"

edit_item: chained_call
         | method_stmt
         | object_def
         | struct_def
         | nested_edit_block
         | assignment

nested_edit_block: "edit" "{" edit_item* "}"

method_stmt: method_invocation ";"

chained_call: NAME ("." method_invocation)+ ";"

method_invocation: NAME "(" [value ("," value)*] ")"

relations_block: "relations" "{" relation_item* "}"
relation_item: relation_method_call
             | relation_fn_call
             | relation_block

relation_method_call: NAME "." NAME "(" [value ("," value)*] ")" ";"
relation_fn_call: NAME "(" NAME "," NAME ["," value] ")" ";"
relation_block: NAME "{" assignment* "}"

constraints_block: "constraints" "{" constraint_item* "}"
constraint_item: constraint_method_call
               | constraint_fn_call
               | constraint_block

constraint_method_call: NAME "." NAME "(" [value ("," value)*] ")" ";"
constraint_fn_call: NAME "(" NAME ["," value]* ")" ";"
constraint_block: NAME "{" assignment* "}"

operations_block: "operations" "{" operation_item* "}"
operation_item: chained_call
              | nested_edit_block
              | NAME ["()"] ";"

value: shape_call
     | struct_call
     | linspace_call
     | summon_call
     | search_call
     | copy_call
     | array_val
     | ESCAPED_STRING
     | SIGNED_NUMBER
     | NAME
     | "true" -> true_val
     | "false" -> false_val

array_val: "[" [value ("," value)*] [","] "]"

NAME: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%import common.CPP_COMMENT
%import common.C_COMMENT

%ignore WS
%ignore CPP_COMMENT
%ignore C_COMMENT
'''
