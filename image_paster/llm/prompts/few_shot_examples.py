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
        search("dense misty pine forest landscape");
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
                search("African elephant full body");
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
                search("large green oak tree");
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

        object wildflowers {
            source {
                search("wildflowers on grass");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = bottom_right;
            standing_on = ground;
            transformation {
                scale = small;
            }
        }
    }

    relations {
        elephant.behind(tree);
        elephant.standing_on(ground);
        tree.standing_on(ground);
        wildflowers.standing_on(ground);
    }

    constraints {
        elephant.must_touch(ground);
        tree.must_touch(ground);
        tree.must_occlude(elephant);
        wildflowers.must_touch(ground);
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
        search("futuristic spaceship bridge interior");
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
                search("cute red panda sitting");
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
                search("wooden dining chair");
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
    {
        "prompt": "two cars on a coastal road",
        "dsl": """// Scene: Two cars on coastal road with copy instruction
scene CoastalCarsScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = car;
    }

    environment {
        search("scenic coastal asphalt road next to ocean");
        type = "coastal_road";
        ground = "asphalt";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        object car {
            source {
                search("a red car on the road");
                viewpoint = side;
                full_body = required;
                isolated = preferred;
            }
            depth = foreground;
            region = left;
            standing_on = ground;
            facing = right;
            transformation {
                scale = large;
                facing = right;
            }
        }

        // Copy car to create second car without re-retrieving
        object car2 = copy(car) {
            depth = midground;
            region = right;
            standing_on = ground;
        }

        // Freely add contextual object related to scene
        object roadside_sign {
            source {
                search("roadside traffic sign");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = bottom_right;
            standing_on = ground;
            transformation {
                scale = small;
            }
        }
    }

    edits {
        car2.scale(0.7).facing(right);
    }

    relations {
        car2.right_of(car);
        car.standing_on(ground);
        car2.standing_on(ground);
        roadside_sign.standing_on(ground);
    }

    constraints {
        car.must_touch(ground);
        car2.must_touch(ground);
        roadside_sign.must_touch(ground);
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
