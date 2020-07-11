# - Install dependencies
#   $ gem install readline
#   $ gem install nats
#
# - Start a NATS server
#   $ nats-server
#
# - Start the bot
#   $ make macondo_bot && ./bin/bot
#
# - Start the test shell
#   $ ruby test/bot-test.rb

require 'nats/client'
require 'fiber'
require 'readline'

CHAN = 'macondo.bot'

NATS.start do
  Fiber.new do
    puts "Macondo bot test shell"
    while buf = Readline.readline("> ", true)
      buf.strip!
      response = NATS.request(CHAN, buf)
      puts response
      puts
    end
    NATS.stop
  end.resume
end
