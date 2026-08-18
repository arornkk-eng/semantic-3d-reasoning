import playcanvasConfig from '@playcanvas/eslint-config';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import globals from 'globals';

export default [
    ...playcanvasConfig,
    {
        files: ['**/*.ts'],
        languageOptions: {
            parser: tsParser,
            globals: {
                ...globals.browser,
                ...globals.serviceworker,
                BlobPart: 'readonly'
            }
        },
        plugins: {
            '@typescript-eslint': tsPlugin
        },
        settings: {
            'import/resolver': {
                typescript: {}
            }
        },
        rules: {
            ...tsPlugin.configs.recommended.rules,
            '@typescript-eslint/ban-ts-comment': 'off',
            '@typescript-eslint/no-explicit-any': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
            'jsdoc/require-param': 'off',
            'jsdoc/require-param-type': 'off',
            'jsdoc/require-returns': 'off',
            'jsdoc/require-returns-type': 'off',
            'jsdoc/check-tag-names': 'off',
            'lines-between-class-members': 'off',
            // TypeScript resolves browser types; ESLint's base no-undef rule does not.
            'no-undef': 'off',
            // The editor intentionally uses native confirmation dialogs and `void promise`
            // for UI event handlers that must not return a promise to the event system.
            'no-alert': 'off',
            'no-void': 'off',
            'no-await-in-loop': 'off',
            'require-atomic-updates': 'off'
        }
    }, {
        files: ['**/*.mjs'],
        languageOptions: {
            globals: {
                ...globals.node
            }
        },
        rules: {
            'import/no-unresolved': 'off'
        }
    }
];
