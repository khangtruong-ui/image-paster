"""Lark EBNF grammar for C++ style Semantic Scene DSL."""

CPP_SCENE_DSL_GRAMMAR = r'''
start: scene

scene: "scene" [NAME] "{" scene_body* "}"

scene_body: camera_block
          | environment_block
          | objects_block
          | relations_block
          | constraints_block
          | operations_block

camera_block: "camera" "{" assignment* "}"

environment_block: "environment" "{" env_item* "}"
env_item: assignment
        | lighting_block

lighting_block: "lighting" "{" assignment* "}"

objects_block: "objects" "{" object_def* "}"
object_def: "object" NAME "{" object_item* "}"

object_item: source_block
           | appearance_block
           | transform_block
           | lighting_block
           | assignment

source_block: "source" "{" assignment* "}"
appearance_block: "appearance" "{" assignment* "}"
transform_block: "transformation" "{" assignment* "}"

assignment: NAME "=" value ";"

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
operation_item: NAME ["()"] ";"

value: ESCAPED_STRING
     | SIGNED_NUMBER
     | NAME
     | "true" -> true_val
     | "false" -> false_val

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
