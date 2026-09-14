"""Few-shot examples for LLM Scene Planner."""

FEW_SHOT_EXAMPLES = [
    {
        "prompt": "an elephant standing behind a tree in a forest",
        "dsl": """// Chain of Thought:
// 1. Scene & Lighting Analysis: Forest environment with gentle morning sunlight. Balanced natural temperature.
// 2. Contextual Logic: I believe the scene of a forest should have wildflowers on the ground, so I freely add wildflowers to the bottom right.
// 3. Occlusion & Composition: The tree is in the foreground on the left, and the elephant is in midground behind the tree, requiring tree.must_occlude(elephant).

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
        "dsl": """// Chain of Thought:
// 1. Scene & Lighting Analysis: Futuristic spaceship interior. Lighting is overhead and cool.
// 2. Object Logic: A red panda is sitting on a wooden chair. The chair stands on the metal deck, and the red panda is placed on the chair.
// 3. Physical Constraints: The chair must physically support the red panda and be larger than it.

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
        "dsl": """// Chain of Thought:
// 1. Scene Analysis: Coastal road scene with two vehicles.
// 2. Copy Logic: Rather than retrieving a second car from scratch, I copy the primary car (`car2 = copy(car);`) to keep consistent visual style.
// 3. Transformation & Context: Scale car2 to 0.7 to place it further along the road in midground. Add a roadside sign for contextual realism.

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
    {
        "prompt": "a vintage car driving along a dark mountain road at night",
        "dsl": """// Chain of Thought:
// 1. Scene Analysis: A vintage car on a winding mountain road at night.
// 2. Lighting & Atmosphere: It is a dark scene so I should make the trees dim, setting low brightness and cool night tones.
// 3. Contextual Reasoning: I believe the scene of a mountain should have trees, so I add a pine tree on the roadside.
// 4. Copy & Variation Logic: I believe the scene of a mountain should have trees, so I copy this tree (`object tree2 = copy(tree) { ... }`) and scale it down to 0.65 in midground.
// 5. Visual Consistency: Since it is a dark scene, both trees have dimmed brightness to seamlessly match the night atmosphere.

scene DarkMountainRoadScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = car;
    }

    environment {
        search("dark mountain road at night landscape");
        type = "mountain";
        ground = "asphalt";
        lighting {
            direction = overhead;
            intensity = soft;
            temperature = cool;
        }
    }

    objects {
        object car {
            source {
                search("a vintage car on the road");
                viewpoint = side;
                full_body = required;
                isolated = preferred;
            }
            depth = foreground;
            region = center;
            standing_on = ground;
            facing = right;
            appearance {
                brightness = -0.15;
            }
            transformation {
                scale = large;
                facing = right;
            }
        }

        object tree {
            source {
                search("pine tree");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = left;
            standing_on = ground;
            appearance {
                brightness = -0.25;
            }
            transformation {
                scale = large;
            }
        }

        // I believe the scene of a mountain should have trees, so I copy this tree:
        object tree2 = copy(tree) {
            depth = midground;
            region = right;
            standing_on = ground;
            appearance {
                brightness = -0.3;
            }
        }
    }

    edits {
        tree2.scale(0.65);
    }

    relations {
        car.standing_on(ground);
        tree.standing_on(ground);
        tree2.standing_on(ground);
        tree.left_of(car);
        tree2.right_of(car);
    }

    constraints {
        car.must_touch(ground);
        tree.must_touch(ground);
        tree2.must_touch(ground);
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
        "prompt": "a row of yellow sunflowers along a stone path in a garden",
        "dsl": """// Chain of Thought:
// 1. Scene & Atmosphere: Sunny garden scene with blooming foliage and warm natural sunlight.
// 2. Object Logic & Adjective Rule: The primary subject is yellow sunflowers. We MUST avoid solitary adjectives like "yellow"; instead use the noun phrase "yellow_sunflower" with query search("yellow sunflower blooming").
// 3. Linspace Duplication: The user requested a "row of sunflowers", so we use the linspace function `linspace(yellow_sunflower, 5)` to duplicate and arrange them along the foreground path.

scene SunflowerGardenScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = sunflowers;
    }

    environment {
        search("lush blooming flower garden with stone path");
        type = "garden";
        ground = "stone_path";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        object yellow_sunflower {
            source {
                search("yellow sunflower blooming");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = bottom;
            standing_on = ground;
            transformation {
                scale = small;
            }
        }

        // Row of sunflowers using linspace:
        object sunflowers = linspace(yellow_sunflower, 5) {
            depth = foreground;
            region = bottom;
            standing_on = ground;
        }
    }

    relations {
        sunflowers.standing_on(ground);
    }

    constraints {
        sunflowers.must_touch(ground);
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
        "prompt": "a circle of glowing candles around a floating crystal in a dark chamber",
        "dsl": """// Chain of Thought:
// 1. Scene Analysis: Dark mystical chamber with cool lighting and a glowing focal point.
// 2. Object Hierarchy: A central crystal floating in the air, surrounded by candles in a circular arrangement.
// 3. Summon Duplication: To form the circle of candles, we use the summon function `summon(candle, 6)` which arranges copies evenly in a 3D perspective circle on the ground.

scene CrystalCandleChamberScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = crystal;
    }

    environment {
        search("dark ancient stone chamber interior");
        type = "room";
        ground = "stone_floor";
        lighting {
            direction = overhead;
            intensity = soft;
            temperature = cool;
        }
    }

    objects {
        object crystal {
            source {
                search("glowing magical crystal floating");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = center;
            appearance {
                brightness = 0.2;
            }
            transformation {
                scale = medium;
            }
        }

        object candle {
            source {
                search("lit wax candle with flame");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            region = center;
            standing_on = ground;
            transformation {
                scale = tiny;
            }
        }

        // Circle of candles using summon:
        object candle_ring = summon(candle, 6) {
            depth = foreground;
            region = center;
            standing_on = ground;
        }
    }

    relations {
        candle_ring.standing_on(ground);
    }

    constraints {
        candle_ring.must_touch(ground);
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
        "prompt": "a line of men each holding a red rose in a stone courtyard",
        "dsl": """// Chain of Thought:
// 1. Scene Analysis: Historic stone courtyard with soft afternoon sunlight.
// 2. Composite Object & Adjective Rule: Each man holds a red rose. NEVER write object "red"; use concrete entity "red_rose" with query search("red rose flower").
// 3. Nested Struct & Linspace: We model "a line of men each holding a flower" using the struct syntax nested inside linspace: `linspace(struct(man, red_rose), 5)`. The engine composites the rose onto the man first, then duplicates the composite entity into a row.

scene CourtyardMenRosesScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = men_with_roses;
    }

    environment {
        search("historic European stone courtyard");
        type = "city";
        ground = "cobblestone";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        object man {
            source {
                search("a man in formal attire standing");
                viewpoint = frontal;
                full_body = required;
                isolated = preferred;
            }
            depth = foreground;
            region = bottom;
            standing_on = ground;
            transformation {
                scale = medium;
            }
        }

        object red_rose {
            source {
                search("a red rose flower");
                viewpoint = frontal;
                isolated = preferred;
            }
            depth = foreground;
            transformation {
                scale = small;
            }
        }

        // Nested struct call inside linspace: creates composite man+rose and lays out in a row
        object men_with_roses = linspace(struct(man, red_rose), 5) {
            depth = foreground;
            region = bottom;
            standing_on = ground;
        }
    }

    relations {
        men_with_roses.standing_on(ground);
    }

    constraints {
        men_with_roses.must_touch(ground);
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

