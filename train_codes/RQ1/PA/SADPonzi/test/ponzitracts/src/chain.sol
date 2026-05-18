pragma solidity ^0.4.24;

contract Ponzi {
    uint Idx = 0;
    struct User {
        address to;
        uint bonus;
    }
    User[] Chain;
    
    function() payable {
        Chain.push(User(msg.sender, msg.value* 3));

        while (this.balance > Chain[Idx].bonus) {
            uint _val = Chain[Idx].bonus;
            Chain[Idx].to.transfer(_val);
            Idx += 1;
        }
    }
}