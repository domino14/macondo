/* global jQuery, _, console, JSON */
(function($, _) {
  "use strict";
  // On DOM initialize:
  $(function() {
    var $results;

    var defaultArgs = {
      'AnagramService.Anagram': {
        mode: 'build',
        letters: 'AEROLITH',
        lexicon: 'America'
      },
      'GaddagService.Generate': {
        filename: '/Users/Cesar/coding/ujamaa/words/OWL2.txt',
        minimize: true
      },
      'GaddagService.GenerateDawg': {
        filename: '/Users/Cesar/coding/ujamaa/words/OWL2.txt',
        minimize: true
      },
      'AnagramService.BlankChallenge': {
        wordLength: 7,
        numQuestions: 25,
        lexicon: 'America',
        maxSolutions: 10,
        num2Blanks: 2
      },
      'AnagramService.BuildChallenge': {
        wordLength: 7,
        minWordLength: 4,
        requireLengthSolution: true,
        lexicon: 'America',
        minSolutions: 30,
        maxSolutions: 100
      }
    };

    $('#input-method-name').change(function() {
      var method = $('#input-method-name').val();
      $('#input-args').val(JSON.stringify(defaultArgs[method]));
    });

    $results = $('#textarea-results');
    $.jsonRPC.setup({
      endPoint: '/rpc'
    });
    $('#submit-rpc').click(function() {
      var methodName, args;
      methodName = $('#input-method-name').val();
      args = $('#input-args').val();
      try {
        args = JSON.parse(args);
      } catch (e) {
        $results.val("Could not parse arguments: " + e);
        return;
      }
      if (!_.isObject(args)) {
        $results.val("You must enter an object as an argument.");
        return;
      }
      $.jsonRPC.request(methodName, {
        params: args,
        success: function(result) {
          $results.val("Success! Result was: " + JSON.stringify(result.result));
          console.log('Success! Result was:', result);
        },
        error: function(result) {
          $results.val("Failure :( Result was: " + result.error.message);
          console.log('Failure. Result:', result);
        }
      });
    });

    // Trigger change event.
    $('#input-method-name').change();

  });

}(jQuery, _));