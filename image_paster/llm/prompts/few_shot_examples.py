"""Few-shot examples for LLM Scene Planner."""

FEW_SHOT_EXAMPLES = [
    {
        "prompt": "an elephant standing behind a tree in a forest",
        "dsl": """// Scene: Elephant behind tree in forest
scene ElephantForestScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = elephant;
    }

    environment {
        type = "forest";
        ground = "grassy";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        object elephant {
            source {
                search("african elephant");
                viewpoint = side;
                full_body = required;
                isolated = preferred;
            }
            depth = midground;
            region = center;
            standing_on = ground;
            facing = right;
            appearance {
                color = "gray";
                lighting = inherit_scene;
            }
            transformation {
                scale = large;
                facing = right;
            }
        }

        object tree {
            source {
                search("tree isolated white background");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = left;
            standing_on = ground;
            transformation {
                scale = large;
            }
        }
    }

    relations {
        elephant.behind(tree);
        elephant.standing_on(ground);
        tree.standing_on(ground);
    }

    constraints {
        elephant.must_touch(ground);
        tree.must_touch(ground);
        tree.must_occlude(elephant);
    }

    operations {
        retrieve;
        segment;
        solve_layout;
        compose;
        blend;
        verify;
    }
}
""",
    },
    {
        "prompt": "a red panda sitting on a wooden chair inside a spaceship",
        "dsl": """// Scene: Red panda sitting on wooden chair inside spaceship
scene RedPandaSpaceshipScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = red_panda;
    }

    environment {
        type = "spaceship_interior";
        ground = "metal_deck";
        lighting {
            direction = overhead;
            intensity = medium;
            temperature = cool;
        }
    }

    objects {
        object red_panda {
            source {
                search("red panda full body");
                viewpoint = frontal;
                full_body = required;
                isolated = preferred;
            }
            depth = foreground;
            region = center;
            standing_on = wooden_chair;
            facing = toward_camera;
            appearance {
                color = "reddish_brown";
                lighting = inherit_scene;
            }
            transformation {
                scale = medium;
            }
        }

        object wooden_chair {
            source {
                search("wooden chair isolated white background");
                viewpoint = frontal;
                full_body = required;
                isolated = preferred;
            }
            depth = foreground;
            region = center;
            standing_on = ground;
            transformation {
                scale = medium;
            }
        }
    }

    relations {
        red_panda.standing_on(wooden_chair);
        wooden_chair.standing_on(ground);
    }

    constraints {
        wooden_chair.must_touch(ground);
        red_panda.must_touch(wooden_chair);
        wooden_chair.must_be_larger_than(red_panda);
    }

    operations {
        retrieve;
        segment;
        solve_layout;
        compose;
        blend;
        verify;
    }
}
""",
    },
]
