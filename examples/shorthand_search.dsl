// Scene: Savanna using shorthand search prompts
// Demonstrates elephant = search("red africa elephant"); syntax
scene SavannaScene {
    camera {
        viewpoint = eye_level;
        perspective = natural;
        focus = elephant;
    }

    environment {
        type = "savanna";
        ground = "dry_grass";
        lighting {
            direction = upper_left;
            intensity = medium;
            temperature = warm;
        }
    }

    objects {
        elephant = search("red africa elephant");
        tree = search("acacia tree");
    }

    relations {
        elephant.left_of(tree);
        elephant.standing_on(ground);
        tree.standing_on(ground);
    }

    constraints {
        elephant.must_touch(ground);
        tree.must_touch(ground);
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
