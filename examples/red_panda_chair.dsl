// Scene: A red panda sitting on a wooden chair inside a spaceship
// Written in C++ style Semantic Scene DSL

scene RedPandaSpaceshipScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = red_panda;
    }

    environment {
        type = "spaceship";
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
                viewpoint = frontal;
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
